#include "runtime/error.hpp"

#include <cuda_runtime.h>

#include <sstream>

namespace qw38::runtime {

char const* code_name(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::Cuda:
      return "cuda";
    case ErrorCode::Format:
      return "format";
    case ErrorCode::Overflow:
      return "overflow";
    case ErrorCode::InvalidCapacity:
      return "invalid_capacity";
    case ErrorCode::InvalidPopulatedLength:
      return "invalid_populated_length";
    case ErrorCode::MalformedArtifact:
      return "malformed_artifact";
    case ErrorCode::AllocationFailed:
      return "allocation_failed";
    case ErrorCode::InvalidArgument:
      return "invalid_argument";
    case ErrorCode::Internal:
      return "internal";
  }
  return "unknown";
}

Error make_error(ErrorCode code, std::string_view field, std::string_view detail,
                 std::uint64_t offset) {
  return Error{.code = code,
               .offset = offset,
               .field = std::string(field),
               .detail = std::string(detail)};
}

Error from_cuda(qw38::cuda::Error const& error) {
  ErrorCode code = ErrorCode::Cuda;
  if (error.code == qw38::cuda::ErrorCode::InvalidArgument) {
    code = ErrorCode::InvalidArgument;
  } else if (error.code == qw38::cuda::ErrorCode::Overflow) {
    code = ErrorCode::Overflow;
  } else if (error.cuda_status == static_cast<int>(cudaErrorMemoryAllocation)) {
    code = ErrorCode::AllocationFailed;
  }
  return Error{.code = code,
               .offset = 0,
               .field = error.operation,
               .detail = qw38::cuda::error_message(error),
               .cuda_status = error.cuda_status};
}

Error from_format(qw38::format::FormatError const& error) {
  return Error{.code = ErrorCode::Format,
               .offset = error.offset,
               .field = error.field,
               .detail = qw38::format::error_message(error)};
}

std::string error_message(Error const& error) {
  std::ostringstream out;
  out << code_name(error.code);
  if (!error.field.empty()) {
    out << " field=" << error.field;
  }
  if (error.offset != 0) {
    out << " offset=" << error.offset;
  }
  if (error.cuda_status != 0) {
    out << " cuda_status=" << error.cuda_status;
  }
  if (!error.detail.empty()) {
    out << ' ' << error.detail;
  }
  return out.str();
}

}  // namespace qw38::runtime
