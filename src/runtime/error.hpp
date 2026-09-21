#pragma once

#include "cuda/error.hpp"
#include "format/error.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace qw38::runtime {

enum class ErrorCode : std::uint16_t {
  Cuda = 1,
  Format = 2,
  Overflow = 3,
  InvalidCapacity = 4,
  InvalidPopulatedLength = 5,
  MalformedArtifact = 6,
  AllocationFailed = 7,
  InvalidArgument = 8,
  Internal = 9,
};

struct Error {
  ErrorCode code{};
  std::uint64_t offset{};
  std::string field;
  std::string detail;
  int cuda_status{};

  friend bool operator==(Error const&, Error const&) = default;
};

[[nodiscard]] char const* code_name(ErrorCode code) noexcept;

[[nodiscard]] Error make_error(ErrorCode code, std::string_view field,
                               std::string_view detail = {},
                               std::uint64_t offset = 0);

[[nodiscard]] Error from_cuda(qw38::cuda::Error const& error);
[[nodiscard]] Error from_format(qw38::format::FormatError const& error);

[[nodiscard]] std::string error_message(Error const& error);

}  // namespace qw38::runtime
