#include "qw38/status.h"

#include <utility>

namespace qw38 {

Status::Status(StatusCode code, std::string message) noexcept
    : code_(code), message_(std::move(message)) {}

Status Status::ok() noexcept { return {}; }
bool Status::is_ok() const noexcept { return code_ == StatusCode::kOk; }
StatusCode Status::code() const noexcept { return code_; }
const std::string& Status::message() const noexcept { return message_; }

const char* status_code_name(StatusCode code) noexcept {
  switch (code) {
    case StatusCode::kOk:
      return "ok";
    case StatusCode::kInvalidArgument:
      return "invalid_argument";
    case StatusCode::kIoError:
      return "io_error";
    case StatusCode::kIncompatibleArtifact:
      return "incompatible_artifact";
    case StatusCode::kResourceExhausted:
      return "resource_exhausted";
    case StatusCode::kCancelled:
      return "cancelled";
    case StatusCode::kUnimplemented:
      return "unimplemented";
    case StatusCode::kInternal:
      return "internal";
  }
  return "unknown";
}

}  // namespace qw38
