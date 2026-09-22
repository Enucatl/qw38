#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace qw38::format {

enum class FormatErrorCode : std::uint16_t {
  Truncated = 1,
  BufferTooSmall = 2,
  LeftoverBytes = 3,
  BadMagic = 4,
  UnsupportedContainerVersion = 5,
  UnsupportedManifestVersion = 6,
  UnknownEnum = 7,
  ReservedNonzero = 8,
  Overflow = 9,
  Misaligned = 10,
  InvalidShape = 11,
  InvalidSpan = 12,
  InvalidMapping = 13,
  InvalidQuantizerLayoutPair = 14,
  InvalidStorageQuantizerPair = 15,
  SharedBinding = 16,
  InvalidGraphBinding = 17,
  InvalidStateAllocation = 18,
  InvalidScratchAllocation = 19,
  InvalidPrecisionPolicy = 20,
  LiveStateForbidden = 21,
  InvalidHeader = 22,
  NameTooLong = 23,
  DuplicateId = 24,
  IoFailure = 25,
  DuplicateSpan = 26,
  MissingSpan = 27,
  IncompleteSpan = 28,
  EmptySpan = 29,
  InconsistentInput = 30,
  OverlappingSpan = 31,
  PublishFailed = 32,
  IntegrityDigestMismatch = 33,
  MissingIntegrity = 34,
  TensorNotFound = 35,
  ResourceLimitExceeded = 36,
  AllocationFailure = 37,
};

struct FormatError {
  FormatErrorCode code{};
  std::uint64_t offset{};
  std::string field;
  std::string detail;

  friend bool operator==(FormatError const&, FormatError const&) = default;
};

[[nodiscard]] char const* code_name(FormatErrorCode code) noexcept;

[[nodiscard]] FormatError make_error(FormatErrorCode code, std::uint64_t offset,
                                     std::string_view field,
                                     std::string_view detail = {});

[[nodiscard]] std::string error_message(FormatError const& error);

}  // namespace qw38::format
