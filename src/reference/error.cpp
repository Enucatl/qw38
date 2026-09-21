#include "reference/error.hpp"

#include <sstream>

namespace qw38::reference {

char const* code_name(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::InvalidArgument:
      return "invalid_argument";
    case ErrorCode::InvalidIndex:
      return "invalid_index";
    case ErrorCode::InvalidShape:
      return "invalid_shape";
  }
  return "unknown";
}

Error make_error(ErrorCode code, std::string_view field,
                 std::string_view detail) {
  return Error{.code = code,
               .field = std::string(field),
               .detail = std::string(detail)};
}

std::string error_message(Error const& error) {
  std::ostringstream out;
  out << code_name(error.code);
  if (!error.field.empty()) {
    out << " field=" << error.field;
  }
  if (!error.detail.empty()) {
    out << ' ' << error.detail;
  }
  return out.str();
}

}  // namespace qw38::reference
