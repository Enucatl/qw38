#include "format/error.hpp"

#include <sstream>

namespace qw38::format {

char const* code_name(FormatErrorCode code) noexcept {
  switch (code) {
    case FormatErrorCode::Truncated:
      return "truncated";
    case FormatErrorCode::BufferTooSmall:
      return "buffer_too_small";
    case FormatErrorCode::LeftoverBytes:
      return "leftover_bytes";
    case FormatErrorCode::BadMagic:
      return "bad_magic";
    case FormatErrorCode::UnsupportedContainerVersion:
      return "unsupported_container_version";
    case FormatErrorCode::UnsupportedManifestVersion:
      return "unsupported_manifest_version";
    case FormatErrorCode::UnknownEnum:
      return "unknown_enum";
    case FormatErrorCode::ReservedNonzero:
      return "reserved_nonzero";
    case FormatErrorCode::Overflow:
      return "overflow";
    case FormatErrorCode::Misaligned:
      return "misaligned";
    case FormatErrorCode::InvalidShape:
      return "invalid_shape";
    case FormatErrorCode::InvalidSpan:
      return "invalid_span";
    case FormatErrorCode::InvalidMapping:
      return "invalid_mapping";
    case FormatErrorCode::InvalidQuantizerLayoutPair:
      return "invalid_quantizer_layout_pair";
    case FormatErrorCode::InvalidStorageQuantizerPair:
      return "invalid_storage_quantizer_pair";
    case FormatErrorCode::SharedBinding:
      return "shared_binding";
    case FormatErrorCode::InvalidGraphBinding:
      return "invalid_graph_binding";
    case FormatErrorCode::InvalidStateAllocation:
      return "invalid_state_allocation";
    case FormatErrorCode::InvalidScratchAllocation:
      return "invalid_scratch_allocation";
    case FormatErrorCode::InvalidPrecisionPolicy:
      return "invalid_precision_policy";
    case FormatErrorCode::LiveStateForbidden:
      return "live_state_forbidden";
    case FormatErrorCode::InvalidHeader:
      return "invalid_header";
    case FormatErrorCode::NameTooLong:
      return "name_too_long";
    case FormatErrorCode::DuplicateId:
      return "duplicate_id";
  }
  return "unknown";
}

FormatError make_error(FormatErrorCode code, std::uint64_t offset,
                       std::string_view field, std::string_view detail) {
  return FormatError{
      .code = code,
      .offset = offset,
      .field = std::string(field),
      .detail = std::string(detail),
  };
}

std::string error_message(FormatError const& error) {
  std::ostringstream out;
  out << code_name(error.code) << " at offset " << error.offset << " field '"
      << error.field << "'";
  if (!error.detail.empty()) {
    out << ": " << error.detail;
  }
  return out.str();
}

}  // namespace qw38::format
