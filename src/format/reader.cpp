#include "format/reader.hpp"

#include "format/constants.hpp"
#include "format/layout.hpp"
#include "format/sha256.hpp"
#include "format/unpack.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>

namespace qw38::format {
namespace {

FormatError io_error(std::uint64_t offset, std::string_view field,
                     std::string_view detail) {
  return make_error(FormatErrorCode::IoFailure, offset, field, detail);
}

class UniqueFd {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(int fd) noexcept : fd_(fd) {}
  ~UniqueFd() { reset(); }

  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset();
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }

  UniqueFd(UniqueFd const&) = delete;
  UniqueFd& operator=(UniqueFd const&) = delete;

  [[nodiscard]] int get() const noexcept { return fd_; }

  void reset() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

 private:
  int fd_{-1};
};

class MappedFile {
 public:
  MappedFile() noexcept = default;

  MappedFile(MappedFile&& other) noexcept
      : fd_(std::move(other.fd_)), addr_(other.addr_), size_(other.size_) {
    other.addr_ = MAP_FAILED;
    other.size_ = 0;
  }

  MappedFile& operator=(MappedFile&& other) noexcept {
    if (this != &other) {
      unmap();
      fd_ = std::move(other.fd_);
      addr_ = other.addr_;
      size_ = other.size_;
      other.addr_ = MAP_FAILED;
      other.size_ = 0;
    }
    return *this;
  }

  MappedFile(MappedFile const&) = delete;
  MappedFile& operator=(MappedFile const&) = delete;

  ~MappedFile() { unmap(); }

  [[nodiscard]] static std::expected<MappedFile, FormatError> open(
      std::filesystem::path const& path) {
    int raw = -1;
    do {
      raw = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    } while (raw < 0 && errno == EINTR);
    if (raw < 0) {
      return std::unexpected(io_error(0, "path", std::strerror(errno)));
    }
    UniqueFd fd{raw};

    struct stat st {};
    if (::fstat(fd.get(), &st) != 0) {
      return std::unexpected(io_error(0, "path", std::strerror(errno)));
    }
    if (!S_ISREG(st.st_mode)) {
      return std::unexpected(io_error(0, "path", "artifact is not a regular file"));
    }
    if (st.st_size < 0) {
      return std::unexpected(io_error(0, "path", "negative file size"));
    }
    auto const size64 = static_cast<std::uint64_t>(st.st_size);
    if (size64 > std::numeric_limits<std::size_t>::max()) {
      return std::unexpected(make_error(FormatErrorCode::Overflow, 0, "path",
                                        "file size exceeds size_t"));
    }
    auto const size = static_cast<std::size_t>(size64);
    if (size < kHeaderSizeV0) {
      return std::unexpected(make_error(FormatErrorCode::Truncated, 0, "header",
                                        "file is smaller than the 64-byte header"));
    }

    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.get(), 0);
    if (addr == MAP_FAILED) {
      return std::unexpected(io_error(0, "path", std::strerror(errno)));
    }

    MappedFile mapped;
    mapped.fd_ = std::move(fd);
    mapped.addr_ = addr;
    mapped.size_ = size;
    return mapped;
  }

  [[nodiscard]] std::span<std::byte const> bytes() const noexcept {
    if (addr_ == MAP_FAILED) {
      return {};
    }
    return std::span<std::byte const>{static_cast<std::byte const*>(addr_),
                                      size_};
  }

 private:
  void unmap() noexcept {
    if (addr_ != MAP_FAILED) {
      ::munmap(addr_, size_);
      addr_ = MAP_FAILED;
      size_ = 0;
    }
  }

  UniqueFd fd_{};
  void* addr_{MAP_FAILED};
  std::size_t size_{0};
};

TensorRecord const* tensor_by_id(ArtifactSchema const& schema,
                                 std::uint32_t id) noexcept {
  for (auto const& t : schema.tensors) {
    if (t.tensor_id == id) {
      return &t;
    }
  }
  return nullptr;
}

TensorRecord const* tensor_by_name(ArtifactSchema const& schema,
                                   std::string_view name) noexcept {
  for (auto const& t : schema.tensors) {
    if (t.logical_name == name) {
      return &t;
    }
  }
  return nullptr;
}

struct Occupied {
  std::uint64_t offset{};
  std::uint64_t length{};
  std::string_view field;
};

bool spans_overlap(Occupied const& a, Occupied const& b) noexcept {
  if (a.length == 0 || b.length == 0) {
    return false;
  }
  auto const a_end = a.offset + a.length;
  auto const b_end = b.offset + b.length;
  return a.offset < b_end && b.offset < a_end;
}

std::expected<void, FormatError> require_in_file(ByteSpan const& span,
                                                 std::uint64_t file_size,
                                                 std::string_view field) {
  if (span.empty()) {
    if (span.offset != 0) {
      return std::unexpected(make_error(FormatErrorCode::InvalidSpan, span.offset,
                                        field, "empty span must use offset 0"));
    }
    return {};
  }
  return check_range(span.offset, span.length, file_size, field);
}

std::expected<std::span<std::byte const>, FormatError> slice(
    std::span<std::byte const> file, ByteSpan const& region,
    std::string_view field) {
  if (region.empty()) {
    return std::span<std::byte const>{};
  }
  if (auto st = require_in_file(region, file.size(), field); !st) {
    return std::unexpected(st.error());
  }
  if (region.offset > std::numeric_limits<std::size_t>::max() ||
      region.length > std::numeric_limits<std::size_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, region.offset,
                                      field, "span exceeds size_t"));
  }
  return file.subspan(static_cast<std::size_t>(region.offset),
                      static_cast<std::size_t>(region.length));
}

std::expected<void, FormatError> check_occupied(std::vector<Occupied>& occupied) {
  std::sort(occupied.begin(), occupied.end(),
            [](Occupied const& a, Occupied const& b) {
              return a.offset < b.offset;
            });
  for (std::size_t i = 1; i < occupied.size(); ++i) {
    if (spans_overlap(occupied[i - 1], occupied[i])) {
      return std::unexpected(make_error(
          FormatErrorCode::OverlappingSpan, occupied[i].offset,
          occupied[i].field, "header, payload, scale, or manifest spans overlap"));
    }
  }
  return {};
}

std::expected<void, FormatError> validate_file_layout(
    ContainerHeader const& header, ArtifactSchema const& schema,
    std::uint64_t file_size) {
  std::vector<Occupied> occupied;
  occupied.push_back(Occupied{0, kHeaderSizeV0, "header"});
  occupied.push_back(Occupied{header.manifest_offset, header.manifest_length,
                              "manifest"});

  std::unordered_set<std::uint32_t> checked_owners;
  for (auto const& tensor : schema.tensors) {
    auto const owner_id =
        canonical_owner_tensor_id(schema, tensor.tensor_id);
    if (!owner_id) {
      return std::unexpected(owner_id.error());
    }
    if (!checked_owners.insert(*owner_id).second) {
      continue;
    }
    auto const* owner = tensor_by_id(schema, *owner_id);
    if (owner == nullptr) {
      return std::unexpected(make_error(FormatErrorCode::SharedBinding, 0,
                                        "shared.owner",
                                        "canonical owner is missing"));
    }
    if (auto st = require_in_file(owner->payload, file_size, "tensor.payload");
        !st) {
      return st;
    }
    if (auto st = require_in_file(owner->scales, file_size, "tensor.scales");
        !st) {
      return st;
    }
    if (!owner->payload.empty()) {
      occupied.push_back(Occupied{owner->payload.offset, owner->payload.length,
                                  "tensor.payload"});
    }
    if (!owner->scales.empty()) {
      occupied.push_back(Occupied{owner->scales.offset, owner->scales.length,
                                  "tensor.scales"});
    }
  }

  for (auto const& rec : schema.integrity) {
    if (auto st = require_in_file(rec.region, file_size, "integrity.region");
        !st) {
      return st;
    }
  }
  return check_occupied(occupied);
}

std::expected<Hash256, FormatError> require_manifest_integrity(
    ContainerHeader const& header, ArtifactSchema const& schema) {
  if (header.manifest_length < kIntegrityRecordBytes) {
    return std::unexpected(make_error(
        FormatErrorCode::InvalidSpan, header.manifest_offset, "manifest",
        "manifest is smaller than one integrity record"));
  }
  IntegrityRecord const* manifest_rec = nullptr;
  for (std::size_t i = 0; i < schema.integrity.size(); ++i) {
    auto const& rec = schema.integrity[i];
    if (rec.kind != IntegrityKind::Sha256Manifest) {
      continue;
    }
    if (manifest_rec != nullptr) {
      return std::unexpected(make_error(FormatErrorCode::DuplicateId,
                                        rec.region.offset, "integrity.kind",
                                        "duplicate manifest digest record"));
    }
    manifest_rec = &rec;
    if (i + 1 != schema.integrity.size()) {
      return std::unexpected(make_error(
          FormatErrorCode::InvalidSpan, rec.region.offset, "integrity.kind",
          "manifest digest record must be the final integrity record"));
    }
  }
  if (manifest_rec == nullptr) {
    return std::unexpected(make_error(FormatErrorCode::MissingIntegrity,
                                      header.manifest_offset, "integrity",
                                      "manifest SHA-256 record is required"));
  }
  if (manifest_rec->region.offset != header.manifest_offset ||
      manifest_rec->region.length != header.manifest_length) {
    // Prefix-hashed artifacts from before complete manifest coverage are not
    // accepted by this corrected protocol and must be regenerated.
    return std::unexpected(make_error(
        FormatErrorCode::InvalidSpan, manifest_rec->region.offset,
        "integrity.region",
        "manifest digest must cover the complete manifest"));
  }
  return manifest_rec->digest;
}

std::expected<void, FormatError> require_span_integrity(
    ArtifactSchema const& schema) {
  std::unordered_set<std::uint32_t> payload_seen;
  std::unordered_set<std::uint32_t> scale_seen;
  for (auto const& rec : schema.integrity) {
    if (rec.kind == IntegrityKind::Sha256Manifest) {
      continue;
    }
    auto const* tensor = tensor_by_id(schema, rec.tensor_id);
    if (tensor == nullptr) {
      return std::unexpected(make_error(
          FormatErrorCode::InvalidSpan, rec.region.offset, "integrity.tensor",
          "integrity record references a missing tensor"));
    }
    auto const owner_id =
        canonical_owner_tensor_id(schema, tensor->tensor_id);
    if (!owner_id) {
      return std::unexpected(owner_id.error());
    }
    auto const* owner = tensor_by_id(schema, *owner_id);
    if (owner == nullptr) {
      return std::unexpected(make_error(FormatErrorCode::SharedBinding, 0,
                                        "shared.owner",
                                        "canonical owner is missing"));
    }
    if (rec.kind == IntegrityKind::Sha256PayloadSpan) {
      if (!payload_seen.insert(rec.tensor_id).second) {
        return std::unexpected(make_error(
            FormatErrorCode::DuplicateId, rec.region.offset, "integrity.tensor",
            "duplicate payload digest for tensor"));
      }
      if (rec.region != owner->payload) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidSpan, rec.region.offset, "integrity.region",
            "payload digest region must match the tensor payload span"));
      }
    } else if (rec.kind == IntegrityKind::Sha256ScaleSpan) {
      if (!scale_seen.insert(rec.tensor_id).second) {
        return std::unexpected(make_error(
            FormatErrorCode::DuplicateId, rec.region.offset, "integrity.tensor",
            "duplicate scale digest for tensor"));
      }
      if (rec.region != owner->scales || owner->scales.empty()) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidSpan, rec.region.offset, "integrity.region",
            "scale digest region must match a nonempty tensor scale span"));
      }
    }
  }

  for (auto const& tensor : schema.tensors) {
    if (!payload_seen.contains(tensor.tensor_id)) {
      return std::unexpected(make_error(
          FormatErrorCode::MissingIntegrity, tensor.payload.offset,
          tensor.logical_name, "payload SHA-256 record is required"));
    }
    if (!tensor.scales.empty() && !scale_seen.contains(tensor.tensor_id)) {
      return std::unexpected(make_error(
          FormatErrorCode::MissingIntegrity, tensor.scales.offset,
          tensor.logical_name, "scale SHA-256 record is required"));
    }
    if (tensor.scales.empty() && scale_seen.contains(tensor.tensor_id)) {
      return std::unexpected(make_error(
          FormatErrorCode::InvalidSpan, tensor.scales.offset,
          tensor.logical_name, "unquantized tensor must not have a scale digest"));
    }
  }
  return {};
}

std::expected<void, FormatError> verify_digests(
    std::span<std::byte const> file, ArtifactSchema const& schema) {
  for (auto const& rec : schema.integrity) {
    auto region = slice(file, rec.region, "integrity.region");
    if (!region) {
      return std::unexpected(region.error());
    }
    Hash256 actual{};
    if (rec.kind == IntegrityKind::Sha256Manifest) {
      std::vector<std::byte> canonical(region->begin(), region->end());
      std::fill(canonical.end() - kHashBytes, canonical.end(), std::byte{});
      actual = sha256(canonical);
    } else {
      actual = sha256(*region);
    }
    if (actual != rec.digest) {
      return std::unexpected(make_error(
          FormatErrorCode::IntegrityDigestMismatch, rec.region.offset,
          rec.kind == IntegrityKind::Sha256Manifest ? "manifest"
                                                    : "integrity.digest",
          "SHA-256 does not match the schema-declared region"));
    }
  }
  return {};
}

std::expected<void, FormatError> validate_quantized_payloads(
    std::span<std::byte const> file, ArtifactSchema const& schema) {
  std::unordered_set<std::uint32_t> checked_owners;
  for (auto const& tensor : schema.tensors) {
    auto const owner_id =
        canonical_owner_tensor_id(schema, tensor.tensor_id);
    if (!owner_id) {
      return std::unexpected(owner_id.error());
    }
    if (!checked_owners.insert(*owner_id).second) {
      continue;
    }
    auto const* owner = tensor_by_id(schema, *owner_id);
    if (owner == nullptr) {
      return std::unexpected(make_error(
          FormatErrorCode::SharedBinding, 0, "shared.owner",
          "canonical owner is missing"));
    }
    if (owner->quantizer != LogicalQuantizerId::Q4G64V0 &&
        owner->quantizer != LogicalQuantizerId::Q8G32V0) {
      continue;
    }
    auto codes = slice(file, owner->payload, "tensor.payload");
    if (!codes) {
      return std::unexpected(codes.error());
    }
    auto scales = slice(file, owner->scales, "tensor.scales");
    if (!scales) {
      return std::unexpected(scales.error());
    }
    auto st = validate_cuda_v0(
        owner->quantizer, owner->layout, owner->shape.logical[0],
        owner->shape.logical[1], *codes, *scales, owner->payload.offset,
        owner->scales.offset);
    if (!st) {
      auto error = st.error();
      error.field = owner->logical_name + "." + error.field;
      return std::unexpected(std::move(error));
    }
  }
  return {};
}

struct Validated {
  ContainerHeader header{};
  ArtifactSchema schema{};
  Hash256 manifest_digest{};
};

std::expected<Validated, FormatError> validate_bytes(
    std::span<std::byte const> bytes) {
  if (bytes.size() < kHeaderSizeV0) {
    return std::unexpected(make_error(FormatErrorCode::Truncated, 0, "header",
                                      "file is smaller than the 64-byte header"));
  }

  auto header = decode_header(bytes.first(kHeaderSizeV0));
  if (!header) {
    return std::unexpected(header.error());
  }

  auto const manifest_end = checked_add(header->manifest_offset,
                                        header->manifest_length, 0, "manifest");
  if (!manifest_end) {
    return std::unexpected(manifest_end.error());
  }
  if (*manifest_end > bytes.size()) {
    return std::unexpected(make_error(FormatErrorCode::Truncated,
                                      header->manifest_offset, "manifest",
                                      "file is shorter than the declared manifest"));
  }
  if (*manifest_end < bytes.size()) {
    return std::unexpected(make_error(FormatErrorCode::LeftoverBytes,
                                      *manifest_end, "file",
                                      "trailing bytes after the declared artifact"));
  }

  auto manifest = slice(bytes,
                        ByteSpan{.offset = header->manifest_offset,
                                 .length = header->manifest_length},
                        "manifest");
  if (!manifest) {
    return std::unexpected(manifest.error());
  }
  auto schema = decode_schema(*manifest, header->manifest_offset);
  if (!schema) {
    return std::unexpected(schema.error());
  }

  if (auto st = validate_file_layout(*header, *schema, bytes.size()); !st) {
    return std::unexpected(st.error());
  }
  auto manifest_digest = require_manifest_integrity(*header, *schema);
  if (!manifest_digest) {
    return std::unexpected(manifest_digest.error());
  }
  if (auto st = require_span_integrity(*schema); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = verify_digests(bytes, *schema); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = validate_quantized_payloads(bytes, *schema); !st) {
    return std::unexpected(st.error());
  }

  Validated out{};
  out.header = *header;
  out.schema = std::move(*schema);
  out.manifest_digest = *manifest_digest;
  return out;
}

ArtifactIdentity make_identity(std::filesystem::path path,
                               Validated const& validated,
                               std::uint64_t size_bytes) {
  ArtifactIdentity id{};
  id.path = std::move(path);
  id.manifest_digest = validated.manifest_digest;
  id.size_bytes = size_bytes;
  id.manifest_offset = validated.header.manifest_offset;
  id.manifest_length = validated.header.manifest_length;
  id.compiler = validated.schema.compiler;
  return id;
}

}  // namespace

struct Artifact::Impl {
  ArtifactIdentity identity{};
  ContainerHeader header{};
  ArtifactSchema schema{};
  std::variant<MappedFile, std::vector<std::byte>> storage;

  [[nodiscard]] std::span<std::byte const> bytes() const noexcept {
    if (auto const* mapped = std::get_if<MappedFile>(&storage)) {
      return mapped->bytes();
    }
    return std::get<std::vector<std::byte>>(storage);
  }
};

Artifact::Artifact(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

Artifact::Artifact(Artifact&&) noexcept = default;
Artifact& Artifact::operator=(Artifact&&) noexcept = default;
Artifact::~Artifact() = default;

std::expected<Artifact, FormatError> Artifact::open(
    std::filesystem::path path) try {
  auto mapped = MappedFile::open(path);
  if (!mapped) {
    return std::unexpected(mapped.error());
  }
  auto validated = validate_bytes(mapped->bytes());
  if (!validated) {
    return std::unexpected(validated.error());
  }
  auto impl = std::make_unique<Impl>();
  impl->identity =
      make_identity(std::move(path), *validated, mapped->bytes().size());
  impl->header = validated->header;
  impl->schema = std::move(validated->schema);
  impl->storage = std::move(*mapped);
  return Artifact{std::move(impl)};
} catch (std::bad_alloc const&) {
  return std::unexpected(
      FormatError{.code = FormatErrorCode::AllocationFailure});
} catch (std::length_error const&) {
  return std::unexpected(make_error(
      FormatErrorCode::ResourceLimitExceeded, 0, "artifact",
      "container size is not representable while opening artifact"));
}

std::expected<Artifact, FormatError> Artifact::parse(
    std::vector<std::byte> const& bytes) {
  return parse(std::span<std::byte const>{bytes});
}

std::expected<Artifact, FormatError> Artifact::parse(
    std::vector<std::byte>&& bytes) try {
  auto validated = validate_bytes(bytes);
  if (!validated) {
    return std::unexpected(validated.error());
  }
  auto impl = std::make_unique<Impl>();
  impl->identity = make_identity({}, *validated, bytes.size());
  impl->header = validated->header;
  impl->schema = std::move(validated->schema);
  impl->storage = std::move(bytes);
  return Artifact{std::move(impl)};
} catch (std::bad_alloc const&) {
  return std::unexpected(
      FormatError{.code = FormatErrorCode::AllocationFailure});
} catch (std::length_error const&) {
  return std::unexpected(make_error(
      FormatErrorCode::ResourceLimitExceeded, 0, "artifact",
      "container size is not representable while parsing artifact"));
}

std::expected<Artifact, FormatError> Artifact::parse(
    std::span<std::byte const> bytes) try {
  // Validate the borrowed bytes, including bounded header and manifest sizes,
  // before allocating an owned copy of a potentially large artifact.
  auto validated = validate_bytes(bytes);
  if (!validated) {
    return std::unexpected(validated.error());
  }
  std::vector<std::byte> owned(bytes.begin(), bytes.end());
  auto impl = std::make_unique<Impl>();
  impl->identity = make_identity({}, *validated, owned.size());
  impl->header = validated->header;
  impl->schema = std::move(validated->schema);
  impl->storage = std::move(owned);
  return Artifact{std::move(impl)};
} catch (std::bad_alloc const&) {
  return std::unexpected(
      FormatError{.code = FormatErrorCode::AllocationFailure});
} catch (std::length_error const&) {
  return std::unexpected(make_error(
      FormatErrorCode::ResourceLimitExceeded, 0, "artifact",
      "container size is not representable while parsing artifact"));
}

ArtifactIdentity const& Artifact::identity() const noexcept {
  return impl_->identity;
}

ContainerHeader const& Artifact::header() const noexcept {
  return impl_->header;
}

ArtifactSchema const& Artifact::schema() const noexcept {
  return impl_->schema;
}

CompilerRevision const& Artifact::compiler() const noexcept {
  return impl_->schema.compiler;
}

PrecisionPolicyRecord const& Artifact::precision() const noexcept {
  return impl_->schema.precision;
}

SemanticScope Artifact::scope() const noexcept { return impl_->schema.scope; }

std::span<StateAllocation const> Artifact::state() const noexcept {
  return impl_->schema.state;
}

std::span<ScratchAllocation const> Artifact::scratch() const noexcept {
  return impl_->schema.scratch;
}

std::span<GraphBinding const> Artifact::graph_bindings() const noexcept {
  return impl_->schema.graph_bindings;
}

std::span<SharedBinding const> Artifact::shared_bindings() const noexcept {
  return impl_->schema.shared_bindings;
}

std::span<TensorRecord const> Artifact::tensors() const noexcept {
  return impl_->schema.tensors;
}

TensorRecord const* Artifact::find_tensor(
    std::string_view logical_name) const noexcept {
  return tensor_by_name(impl_->schema, logical_name);
}

TensorRecord const* Artifact::find_tensor(std::uint32_t tensor_id) const noexcept {
  return tensor_by_id(impl_->schema, tensor_id);
}

std::span<std::byte const> Artifact::mapped_bytes() const noexcept {
  return impl_->bytes();
}

std::span<std::byte const> Artifact::view(ByteSpan region) const noexcept {
  if (region.empty()) {
    return {};
  }
  return mapped_bytes().subspan(static_cast<std::size_t>(region.offset),
                                static_cast<std::size_t>(region.length));
}

std::expected<std::span<std::byte const>, FormatError> Artifact::span(
    std::string_view logical_name, SpanKind kind) const {
  auto const* tensor = find_tensor(logical_name);
  if (tensor == nullptr) {
    return std::unexpected(make_error(FormatErrorCode::TensorNotFound, 0,
                                      logical_name,
                                      "no tensor with this logical identity"));
  }
  return view(kind == SpanKind::Payload ? tensor->payload : tensor->scales);
}

std::expected<std::span<std::byte const>, FormatError> Artifact::payload(
    std::string_view logical_name) const {
  return span(logical_name, SpanKind::Payload);
}

std::expected<std::span<std::byte const>, FormatError> Artifact::scales(
    std::string_view logical_name) const {
  return span(logical_name, SpanKind::Scales);
}

}  // namespace qw38::format
