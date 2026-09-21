#include "cuda/error.hpp"

#include <sstream>

namespace qw38::cuda {

char const* code_name(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::Status:
      return "cuda_status";
    case ErrorCode::InvalidArgument:
      return "invalid_argument";
    case ErrorCode::Overflow:
      return "overflow";
  }
  return "unknown";
}

Error from_status(cudaError_t status, std::string_view operation,
                  std::source_location loc) {
  std::ostringstream detail;
  detail << cudaGetErrorString(status) << " (" << loc.file_name() << ':'
         << loc.line() << ')';
  return Error{
      .code = ErrorCode::Status,
      .cuda_status = static_cast<int>(status),
      .operation = std::string(operation),
      .detail = detail.str(),
  };
}

Error make_error(ErrorCode code, std::string_view operation,
                 std::string_view detail) {
  return Error{
      .code = code,
      .cuda_status = static_cast<int>(cudaSuccess),
      .operation = std::string(operation),
      .detail = std::string(detail),
  };
}

std::string error_message(Error const& error) {
  std::ostringstream out;
  out << code_name(error.code);
  if (!error.operation.empty()) {
    out << " op=" << error.operation;
  }
  if (error.cuda_status != static_cast<int>(cudaSuccess)) {
    out << " status=" << error.cuda_status;
  }
  if (!error.detail.empty()) {
    out << ' ' << error.detail;
  }
  return out.str();
}

}  // namespace qw38::cuda
