#include "compiler/error.hpp"

#include <sstream>

namespace qw38::compiler {

char const* code_name(CompilerErrorCode code) noexcept {
  switch (code) {
    case CompilerErrorCode::IoFailure:
      return "io_failure";
    case CompilerErrorCode::InvalidJson:
      return "invalid_json";
    case CompilerErrorCode::InvalidConfig:
      return "invalid_config";
    case CompilerErrorCode::ArchitectureMismatch:
      return "architecture_mismatch";
    case CompilerErrorCode::MissingTensor:
      return "missing_tensor";
    case CompilerErrorCode::ExtraTensor:
      return "extra_tensor";
    case CompilerErrorCode::DuplicateTensor:
      return "duplicate_tensor";
    case CompilerErrorCode::ShapeMismatch:
      return "shape_mismatch";
    case CompilerErrorCode::DtypeMismatch:
      return "dtype_mismatch";
    case CompilerErrorCode::UnexpectedSharing:
      return "unexpected_sharing";
    case CompilerErrorCode::Nonfinite:
      return "nonfinite";
    case CompilerErrorCode::HashMismatch:
      return "hash_mismatch";
    case CompilerErrorCode::Format:
      return "format";
    case CompilerErrorCode::Internal:
      return "internal";
  }
  return "unknown";
}

CompilerError make_error(CompilerErrorCode code, std::string_view field,
                         std::string_view detail, std::uint64_t offset) {
  return CompilerError{.code = code,
                       .offset = offset,
                       .field = std::string(field),
                       .detail = std::string(detail)};
}

CompilerError from_format(qw38::format::FormatError const& error) {
  return CompilerError{.code = CompilerErrorCode::Format,
                       .offset = error.offset,
                       .field = error.field,
                       .detail = qw38::format::error_message(error)};
}

std::string error_message(CompilerError const& error) {
  std::ostringstream out;
  out << code_name(error.code);
  if (!error.field.empty()) {
    out << " field=" << error.field;
  }
  if (error.offset != 0) {
    out << " offset=" << error.offset;
  }
  if (!error.detail.empty()) {
    out << " " << error.detail;
  }
  return out.str();
}

}  // namespace qw38::compiler
