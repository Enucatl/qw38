#pragma once

#include "format/error.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace qw38::compiler {

enum class CompilerErrorCode : std::uint16_t {
  IoFailure = 1,
  InvalidJson = 2,
  InvalidConfig = 3,
  ArchitectureMismatch = 4,
  MissingTensor = 5,
  ExtraTensor = 6,
  DuplicateTensor = 7,
  ShapeMismatch = 8,
  DtypeMismatch = 9,
  UnexpectedSharing = 10,
  Nonfinite = 11,
  HashMismatch = 12,
  Format = 13,
  Internal = 14,
};

struct CompilerError {
  CompilerErrorCode code{};
  std::uint64_t offset{};
  std::string field;
  std::string detail;

  friend bool operator==(CompilerError const&, CompilerError const&) = default;
};

[[nodiscard]] char const* code_name(CompilerErrorCode code) noexcept;

[[nodiscard]] CompilerError make_error(CompilerErrorCode code,
                                       std::string_view field,
                                       std::string_view detail = {},
                                       std::uint64_t offset = 0);

[[nodiscard]] CompilerError from_format(qw38::format::FormatError const& error);

[[nodiscard]] std::string error_message(CompilerError const& error);

}  // namespace qw38::compiler
