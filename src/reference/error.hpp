#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace qw38::reference {

enum class ErrorCode : std::uint16_t {
  InvalidArgument = 1,
  InvalidIndex = 2,
  InvalidShape = 3,
};

struct Error {
  ErrorCode code{};
  std::string field;
  std::string detail;

  friend bool operator==(Error const&, Error const&) = default;
};

[[nodiscard]] char const* code_name(ErrorCode code) noexcept;

[[nodiscard]] Error make_error(ErrorCode code, std::string_view field,
                               std::string_view detail = {});

[[nodiscard]] std::string error_message(Error const& error);

}  // namespace qw38::reference
