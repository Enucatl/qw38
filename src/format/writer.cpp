#include "format/writer.hpp"

#include "format/constants.hpp"
#include "format/layout.hpp"
#include "format/sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <new>
#include <string>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/stat.h>

namespace qw38::format {
namespace {

ssize_t posix_pwrite(void*, int fd, void const* data, std::size_t size,
                     off_t offset) {
  return ::pwrite(fd, data, size, offset);
}
int posix_fsync(void*, int fd) { return ::fsync(fd); }
int posix_close(void*, int fd) { return ::close(fd); }
int posix_rename(void*, char const* from, char const* to) {
  return ::rename(from, to);
}
int posix_remove(void*, char const* path) { return ::unlink(path); }
WriterFilesystem const kPosixFilesystem{
    .pwrite = posix_pwrite,
    .fsync = posix_fsync,
    .close = posix_close,
    .rename = posix_rename,
    .remove = posix_remove,
};

FormatError io_error(std::uint64_t offset, std::string_view field,
                     std::string_view detail) {
  return make_error(FormatErrorCode::IoFailure, offset, field, detail);
}

class UniqueFd {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(int fd, WriterFilesystem const& filesystem = kPosixFilesystem) noexcept
      : fd_(fd), filesystem_(&filesystem) {}
  ~UniqueFd() { reset(); }

  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_), filesystem_(other.filesystem_) {
    other.fd_ = -1;
  }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset();
      fd_ = other.fd_;
      filesystem_ = other.filesystem_;
      other.fd_ = -1;
    }
    return *this;
  }

  UniqueFd(UniqueFd const&) = delete;
  UniqueFd& operator=(UniqueFd const&) = delete;

  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }

  void reset() noexcept {
    if (fd_ >= 0) {
      filesystem_->close(filesystem_->context, fd_);
      fd_ = -1;
    }
  }

  [[nodiscard]] int close() noexcept {
    if (fd_ < 0) {
      return 0;
    }
    int const result = filesystem_->close(filesystem_->context, fd_);
    fd_ = -1;
    return result;
  }

  int release() noexcept {
    int const fd = fd_;
    fd_ = -1;
    return fd;
  }

 private:
  int fd_{-1};
  WriterFilesystem const* filesystem_{&kPosixFilesystem};
};

std::expected<void, FormatError> write_at(WriterFilesystem const& filesystem,
                                          int fd, std::uint64_t offset,
                                          std::span<std::byte const> data,
                                          std::string_view field) {
  while (!data.empty()) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      return std::unexpected(make_error(FormatErrorCode::Overflow, offset, field,
                                        "file offset exceeds off_t"));
    }
    ssize_t const n = filesystem.pwrite(filesystem.context, fd, data.data(),
                                        data.size(), static_cast<off_t>(offset));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return std::unexpected(io_error(offset, field, std::strerror(errno)));
    }
    if (n == 0) {
      return std::unexpected(io_error(offset, field, "short write"));
    }
    offset += static_cast<std::uint64_t>(n);
    data = data.subspan(static_cast<std::size_t>(n));
  }
  return {};
}

std::expected<void, FormatError> write_zeros(WriterFilesystem const& filesystem,
                                             int fd, std::uint64_t offset,
                                             std::uint64_t length,
                                             std::string_view field) {
  std::array<std::byte, 256> zeros{};
  while (length > 0) {
    std::size_t const n =
        static_cast<std::size_t>(std::min<std::uint64_t>(length, zeros.size()));
    if (auto st = write_at(filesystem, fd, offset,
                           std::span<std::byte const>{zeros.data(), n}, field);
        !st) {
      return st;
    }
    offset += n;
    length -= n;
  }
  return {};
}

TensorRecord const* find_tensor(ArtifactSchema const& schema,
                                std::uint32_t id) {
  for (auto const& t : schema.tensors) {
    if (t.tensor_id == id) {
      return &t;
    }
  }
  return nullptr;
}

bool is_alias(ArtifactSchema const& schema, std::uint32_t tensor_id) {
  for (auto const& b : schema.shared_bindings) {
    if (b.alias_tensor_id == tensor_id) {
      return true;
    }
  }
  return false;
}

std::expected<std::uint32_t, FormatError> owner_of(
    ArtifactSchema const& schema, std::uint32_t tensor_id) {
  return canonical_owner_tensor_id(schema, tensor_id);
}

bool spans_overlap(ByteSpan const& a, ByteSpan const& b) noexcept {
  if (a.empty() || b.empty()) {
    return false;
  }
  auto const a_end = a.offset + a.length;
  auto const b_end = b.offset + b.length;
  return a.offset < b_end && b.offset < a_end;
}

std::expected<void, FormatError> assign_spans(ArtifactSchema& schema) {
  std::uint64_t cursor = kHeaderSizeV0;
  for (auto& tensor : schema.tensors) {
    if (is_alias(schema, tensor.tensor_id)) {
      continue;
    }
    auto const payload_n = expected_payload_bytes(tensor);
    if (!payload_n) {
      return std::unexpected(payload_n.error());
    }
    auto const scale_n = expected_scale_bytes(tensor);
    if (!scale_n) {
      return std::unexpected(scale_n.error());
    }
    if (*payload_n == 0) {
      return std::unexpected(make_error(FormatErrorCode::InvalidSpan, cursor,
                                        tensor.logical_name,
                                        "payload span must be nonempty"));
    }
    auto aligned = align_up(cursor, kSpanAlignment, cursor, tensor.logical_name);
    if (!aligned) {
      return std::unexpected(aligned.error());
    }
    tensor.payload = ByteSpan{.offset = *aligned, .length = *payload_n};
    auto payload_end =
        checked_add(*aligned, *payload_n, *aligned, "tensor.payload");
    if (!payload_end) {
      return std::unexpected(payload_end.error());
    }
    cursor = *payload_end;
    if (*scale_n == 0) {
      tensor.scales = ByteSpan{};
      continue;
    }
    aligned = align_up(cursor, kSpanAlignment, cursor, tensor.logical_name);
    if (!aligned) {
      return std::unexpected(aligned.error());
    }
    tensor.scales = ByteSpan{.offset = *aligned, .length = *scale_n};
    auto scale_end =
        checked_add(*aligned, *scale_n, *aligned, "tensor.scales");
    if (!scale_end) {
      return std::unexpected(scale_end.error());
    }
    cursor = *scale_end;
  }

  for (auto& tensor : schema.tensors) {
    if (!is_alias(schema, tensor.tensor_id)) {
      continue;
    }
    auto const owner_id = owner_of(schema, tensor.tensor_id);
    if (!owner_id) {
      return std::unexpected(owner_id.error());
    }
    auto const* owner = find_tensor(schema, *owner_id);
    if (owner == nullptr) {
      return std::unexpected(make_error(FormatErrorCode::SharedBinding, 0,
                                        "shared.owner",
                                        "alias owner is missing"));
    }
    tensor.payload = owner->payload;
    tensor.scales = owner->scales;
  }

  auto aligned = align_up(cursor, kSpanAlignment, cursor, "manifest");
  if (!aligned) {
    return std::unexpected(aligned.error());
  }
  (void)aligned;
  return {};
}

std::expected<std::uint64_t, FormatError> payload_cursor_end(
    ArtifactSchema const& schema) {
  std::uint64_t end = kHeaderSizeV0;
  for (auto const& tensor : schema.tensors) {
    if (tensor.payload.empty()) {
      continue;
    }
    auto payload_end = checked_add(tensor.payload.offset, tensor.payload.length,
                                   tensor.payload.offset, "tensor.payload");
    if (!payload_end) {
      return std::unexpected(payload_end.error());
    }
    end = std::max(end, *payload_end);
    if (tensor.scales.empty()) {
      continue;
    }
    auto scale_end = checked_add(tensor.scales.offset, tensor.scales.length,
                                 tensor.scales.offset, "tensor.scales");
    if (!scale_end) {
      return std::unexpected(scale_end.error());
    }
    end = std::max(end, *scale_end);
  }
  return end;
}

std::expected<void, FormatError> check_unique_span_overlap(
    ArtifactSchema const& schema) {
  struct Placed {
    ByteSpan span;
    char const* field;
    std::uint32_t tensor_id;
  };
  std::vector<Placed> placed;
  for (auto const& tensor : schema.tensors) {
    if (is_alias(schema, tensor.tensor_id)) {
      continue;
    }
    if (!tensor.payload.empty()) {
      placed.push_back(Placed{tensor.payload, "tensor.payload", tensor.tensor_id});
    }
    if (!tensor.scales.empty()) {
      placed.push_back(Placed{tensor.scales, "tensor.scales", tensor.tensor_id});
    }
  }
  for (std::size_t i = 0; i < placed.size(); ++i) {
    for (std::size_t j = i + 1; j < placed.size(); ++j) {
      if (spans_overlap(placed[i].span, placed[j].span)) {
        return std::unexpected(make_error(
            FormatErrorCode::OverlappingSpan, placed[i].span.offset,
            placed[i].field, "payload/scale spans overlap"));
      }
    }
  }
  return {};
}

std::expected<void, FormatError> validate_input_span(
    ByteSpan const& span, std::uint64_t expected_length,
    std::string_view field) {
  if (span.length != 0 && span.length != expected_length) {
    return std::unexpected(make_error(
        FormatErrorCode::InconsistentInput, span.offset, field,
        "declared span length does not match layout"));
  }
  if ((span.offset == 0) != (span.length == 0)) {
    return std::unexpected(make_error(
        FormatErrorCode::InvalidSpan, span.offset, field,
        "an unplaced span must use offset and length 0"));
  }
  if (auto st = require_span_alignment(span.offset, span.length, field); !st) {
    return std::unexpected(st.error());
  }
  if (auto end = checked_add(span.offset, span.length, span.offset, field);
      !end) {
    return std::unexpected(end.error());
  }
  return {};
}

std::expected<void, FormatError> validate_input_span_placements(
    ArtifactSchema const& schema) {
  struct Placed {
    ByteSpan span;
    SpanKind kind;
    std::uint32_t owner_tensor_id;
  };

  std::vector<Placed> placed;
  placed.reserve(schema.tensors.size() * 2);
  for (auto const& tensor : schema.tensors) {
    auto const payload_n = expected_payload_bytes(tensor);
    if (!payload_n) {
      return std::unexpected(payload_n.error());
    }
    auto const scale_n = expected_scale_bytes(tensor);
    if (!scale_n) {
      return std::unexpected(scale_n.error());
    }
    if (auto st = validate_input_span(tensor.payload, *payload_n,
                                      "tensor.payload");
        !st) {
      return std::unexpected(st.error());
    }
    if (auto st = validate_input_span(tensor.scales, *scale_n,
                                      "tensor.scales");
        !st) {
      return std::unexpected(st.error());
    }
    auto const owner_id = owner_of(schema, tensor.tensor_id);
    if (!owner_id) {
      return std::unexpected(owner_id.error());
    }
    if (!tensor.payload.empty()) {
      placed.push_back(Placed{tensor.payload, SpanKind::Payload, *owner_id});
    }
    if (!tensor.scales.empty()) {
      placed.push_back(Placed{tensor.scales, SpanKind::Scales, *owner_id});
    }
  }

  for (auto const& binding : schema.shared_bindings) {
    auto const* owner = find_tensor(schema, binding.owner_tensor_id);
    auto const* alias = find_tensor(schema, binding.alias_tensor_id);
    if (owner == nullptr || alias == nullptr) {
      return std::unexpected(make_error(FormatErrorCode::SharedBinding, 0,
                                        "shared.owner",
                                        "alias owner is missing"));
    }
    if (alias->payload != owner->payload || alias->scales != owner->scales) {
      return std::unexpected(make_error(
          FormatErrorCode::SharedBinding, alias->payload.offset,
          "shared.alias", "alias spans must exactly match their owner"));
    }
  }

  for (std::size_t i = 0; i < placed.size(); ++i) {
    for (std::size_t j = i + 1; j < placed.size(); ++j) {
      if (!spans_overlap(placed[i].span, placed[j].span)) {
        continue;
      }
      bool const permitted =
          placed[i].kind == placed[j].kind &&
          placed[i].owner_tensor_id == placed[j].owner_tensor_id &&
          placed[i].span == placed[j].span;
      if (!permitted) {
        return std::unexpected(make_error(
            FormatErrorCode::OverlappingSpan, placed[i].span.offset,
            "tensor.span", "input payload/scale spans overlap"));
      }
    }
  }
  return {};
}

std::expected<ArtifactSchema, FormatError> prepare_schema(
    ArtifactSchema const& input) {
  if (!input.integrity.empty()) {
    return std::unexpected(make_error(
        FormatErrorCode::InconsistentInput, 0, "integrity",
        "writer owns integrity records; input must leave them empty"));
  }
  ArtifactSchema schema = input;
  if (auto st = validate_shared_binding_ownership(schema); !st) {
    return std::unexpected(st.error());
  }
  // Tensor collection order is not semantic.  Canonicalize it before assigning
  // physical spans so equivalent schemas cannot produce different artifacts.
  std::sort(schema.tensors.begin(), schema.tensors.end(),
            [](TensorRecord const& a, TensorRecord const& b) {
              return a.tensor_id < b.tensor_id;
            });
  if (auto st = validate_input_span_placements(schema); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = assign_spans(schema); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = check_unique_span_overlap(schema); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = validate_schema(schema); !st) {
    return std::unexpected(st.error());
  }
  return schema;
}

}  // namespace

struct ArtifactWriter::Impl {
  std::filesystem::path destination;
  std::filesystem::path temp_path;
  ArtifactSchema schema;
  UniqueFd fd;
  WriterFilesystem const* filesystem{&kPosixFilesystem};
  std::uint64_t manifest_offset{};
  bool published{false};
  bool closed{false};

  struct PhysicalSpan {
    std::uint32_t owner_tensor_id{};
    SpanKind kind{SpanKind::Payload};
    ByteSpan region{};
    std::string field;
    std::uint64_t written{};
    bool started{false};
    bool complete{false};
  };

  std::vector<PhysicalSpan> spans;
  std::unordered_map<std::string, std::size_t> payload_by_name;
  std::unordered_map<std::string, std::size_t> scales_by_name;

  ~Impl() {
    if (!published) {
      abandon();
    }
  }

  void abandon() noexcept {
    fd.reset();
    if (!published && !temp_path.empty()) {
      filesystem->remove(filesystem->context, temp_path.c_str());
    }
    closed = true;
  }
};

ArtifactWriter::ArtifactWriter(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ArtifactWriter::ArtifactWriter(ArtifactWriter&&) noexcept = default;
ArtifactWriter& ArtifactWriter::operator=(ArtifactWriter&&) noexcept = default;

ArtifactWriter::~ArtifactWriter() {
  if (impl_ && !impl_->published) {
    impl_->abandon();
  }
}

std::expected<ArtifactWriter, FormatError> ArtifactWriter::create(
    std::filesystem::path const& destination, ArtifactSchema const& schema,
    WriterFilesystem const* filesystem) try {
  if (destination.empty()) {
    return std::unexpected(make_error(FormatErrorCode::InconsistentInput, 0,
                                      "destination",
                                      "destination path is empty"));
  }

  auto prepared = prepare_schema(schema);
  if (!prepared) {
    return std::unexpected(prepared.error());
  }

  auto cursor_end = payload_cursor_end(*prepared);
  if (!cursor_end) {
    return std::unexpected(cursor_end.error());
  }
  auto manifest_off =
      align_up(*cursor_end, kSpanAlignment, *cursor_end, "manifest");
  if (!manifest_off) {
    return std::unexpected(manifest_off.error());
  }

  auto impl = std::make_unique<Impl>();
  impl->filesystem = filesystem == nullptr ? &kPosixFilesystem : filesystem;
  if (impl->filesystem->pwrite == nullptr || impl->filesystem->fsync == nullptr ||
      impl->filesystem->close == nullptr || impl->filesystem->rename == nullptr ||
      impl->filesystem->remove == nullptr) {
    return std::unexpected(make_error(FormatErrorCode::InconsistentInput, 0,
                                      "filesystem", "filesystem operations are incomplete"));
  }
  impl->destination = destination;
  impl->schema = std::move(*prepared);
  impl->manifest_offset = *manifest_off;
  impl->temp_path = impl->destination;
  impl->temp_path += ".tmp.XXXXXX";

  std::error_code ec;
  auto const parent = impl->destination.parent_path();
  if (!parent.empty() && !std::filesystem::is_directory(parent, ec)) {
    return std::unexpected(io_error(0, "destination",
                                    "parent directory does not exist"));
  }

  std::string temp_template = impl->temp_path.string();
  int const raw = ::mkstemp(temp_template.data());
  if (raw < 0) {
    return std::unexpected(
        io_error(0, "temporary", std::strerror(errno)));
  }
  impl->temp_path = std::move(temp_template);
  impl->fd = UniqueFd{raw, *impl->filesystem};
  if (::fcntl(raw, F_SETFD, FD_CLOEXEC) < 0) {
    auto const error = io_error(0, "temporary", std::strerror(errno));
    impl->abandon();
    return std::unexpected(error);
  }

  if (auto st = write_zeros(*impl->filesystem, impl->fd.get(), 0,
                            kHeaderSizeV0, "header"); !st) {
    impl->abandon();
    return std::unexpected(st.error());
  }

  std::unordered_map<std::uint32_t, std::size_t> payload_index;
  std::unordered_map<std::uint32_t, std::size_t> scale_index;
  payload_index.reserve(impl->schema.tensors.size());
  scale_index.reserve(impl->schema.tensors.size());

  for (auto const& tensor : impl->schema.tensors) {
    if (is_alias(impl->schema, tensor.tensor_id)) {
      continue;
    }
    Impl::PhysicalSpan payload{
        .owner_tensor_id = tensor.tensor_id,
        .kind = SpanKind::Payload,
        .region = tensor.payload,
        .field = tensor.logical_name + ".payload",
    };
    payload_index[tensor.tensor_id] = impl->spans.size();
    impl->spans.push_back(std::move(payload));
    if (!tensor.scales.empty()) {
      Impl::PhysicalSpan scales{
          .owner_tensor_id = tensor.tensor_id,
          .kind = SpanKind::Scales,
          .region = tensor.scales,
          .field = tensor.logical_name + ".scales",
      };
      scale_index[tensor.tensor_id] = impl->spans.size();
      impl->spans.push_back(std::move(scales));
    }
  }

  for (auto const& tensor : impl->schema.tensors) {
    auto const owner_id = owner_of(impl->schema, tensor.tensor_id);
    if (!owner_id) {
      impl->abandon();
      return std::unexpected(owner_id.error());
    }
    auto const payload_it = payload_index.find(*owner_id);
    if (payload_it != payload_index.end()) {
      impl->payload_by_name.emplace(tensor.logical_name, payload_it->second);
    }
    auto const scale_it = scale_index.find(*owner_id);
    if (scale_it != scale_index.end()) {
      impl->scales_by_name.emplace(tensor.logical_name, scale_it->second);
    }
  }

  return ArtifactWriter{std::move(impl)};
} catch (std::bad_alloc const&) {
  return std::unexpected(FormatError{.code = FormatErrorCode::AllocationFailure,
                                     .offset = 0,
                                     .field = {},
                                     .detail = {}});
} catch (std::length_error const&) {
  return std::unexpected(make_error(
      FormatErrorCode::ResourceLimitExceeded, 0, "writer",
      "writer state size is not representable while creating artifact"));
}

std::expected<void, FormatError> ArtifactWriter::write_span(
    std::string_view logical_name, SpanKind kind,
    std::span<std::byte const> chunk) try {
  if (!impl_ || impl_->closed || impl_->published) {
    return std::unexpected(make_error(FormatErrorCode::PublishFailed, 0,
                                      "writer",
                                      "writer is closed or already published"));
  }
  if (chunk.empty()) {
    return std::unexpected(make_error(FormatErrorCode::EmptySpan, 0,
                                      logical_name,
                                      "span chunk must be nonempty"));
  }

  auto const& map = (kind == SpanKind::Payload) ? impl_->payload_by_name
                                                : impl_->scales_by_name;
  auto const it = map.find(std::string(logical_name));
  if (it == map.end()) {
    return std::unexpected(make_error(
        FormatErrorCode::InconsistentInput, 0, logical_name,
        kind == SpanKind::Payload
            ? "no payload span is declared for this logical name"
            : "no scale span is declared for this logical name"));
  }
  auto& span = impl_->spans[it->second];
  if (span.complete) {
    return std::unexpected(make_error(FormatErrorCode::DuplicateSpan,
                                      span.region.offset, span.field,
                                      "span was already fully written"));
  }
  auto remaining = span.region.length - span.written;
  if (chunk.size() > remaining) {
    return std::unexpected(make_error(
        FormatErrorCode::InconsistentInput, span.region.offset, span.field,
        "chunk exceeds remaining span length"));
  }
  auto const offset = span.region.offset + span.written;
  if (auto st = write_at(*impl_->filesystem, impl_->fd.get(), offset, chunk,
                         span.field); !st) {
    return st;
  }
  span.written += chunk.size();
  span.started = true;
  if (span.written == span.region.length) {
    span.complete = true;
  }
  return {};
} catch (std::bad_alloc const&) {
  if (impl_ && !impl_->published) {
    impl_->abandon();
  }
  return std::unexpected(FormatError{.code = FormatErrorCode::AllocationFailure,
                                     .offset = 0,
                                     .field = {},
                                     .detail = {}});
} catch (std::length_error const&) {
  if (impl_ && !impl_->published) {
    impl_->abandon();
  }
  return std::unexpected(make_error(
      FormatErrorCode::ResourceLimitExceeded, 0, "writer",
      "writer state size is not representable while writing span"));
}

std::expected<ArtifactIdentity, FormatError> ArtifactWriter::finalize() try {
  if (!impl_ || impl_->closed || impl_->published) {
    return std::unexpected(make_error(FormatErrorCode::PublishFailed, 0,
                                      "writer",
                                      "writer is closed or already published"));
  }

  auto fail = [&](FormatError err) -> std::expected<ArtifactIdentity, FormatError> {
    impl_->abandon();
    return std::unexpected(std::move(err));
  };

  for (auto const& span : impl_->spans) {
    if (!span.complete) {
      auto const code = span.started ? FormatErrorCode::IncompleteSpan
                                     : FormatErrorCode::MissingSpan;
      return fail(make_error(code, span.region.offset, span.field,
                             span.started ? "span was only partially written"
                                          : "required span was not written"));
    }
  }

  IntegrityRecord manifest_rec{};
  manifest_rec.kind = IntegrityKind::Sha256Manifest;
  impl_->schema.integrity = {manifest_rec};

  auto encoded_n = encoded_size(impl_->schema);
  if (!encoded_n) {
    return fail(encoded_n.error());
  }
  if (*encoded_n < kIntegrityRecordBytes) {
    return fail(make_error(FormatErrorCode::InvalidSpan, 0, "manifest",
                           "encoded manifest is smaller than one record"));
  }
  impl_->schema.integrity.back().region = ByteSpan{
      .offset = impl_->manifest_offset,
      .length = static_cast<std::uint64_t>(*encoded_n),
  };

  std::vector<std::byte> manifest(*encoded_n);
  if (auto st = encode(impl_->schema, manifest); !st) {
    return fail(st.error());
  }
  // The final integrity digest is self-referential.  Encode its zero-initialized
  // bytes while hashing so the record's kind, tensor id, and region remain
  // covered by the manifest digest without changing the fixed-width wire record.
  Hash256 const digest =
      sha256(std::span<std::byte const>{manifest.data(), manifest.size()});
  impl_->schema.integrity.back().digest = digest;
  if (auto st = encode(impl_->schema, manifest); !st) {
    return fail(st.error());
  }

  if (auto st = validate_schema(impl_->schema); !st) {
    return fail(st.error());
  }
  ContainerHeader header{};
  header.manifest_offset = impl_->manifest_offset;
  header.manifest_length = static_cast<std::uint64_t>(*encoded_n);
  if (auto st = validate_header(header); !st) {
    return fail(st.error());
  }

  struct PadRegion {
    std::uint64_t offset;
    std::uint64_t length;
  };
  std::vector<PadRegion> pads;
  std::vector<ByteSpan> occupied;
  occupied.reserve(impl_->spans.size());
  for (auto const& span : impl_->spans) {
    occupied.push_back(span.region);
  }
  std::sort(occupied.begin(), occupied.end(),
            [](ByteSpan const& a, ByteSpan const& b) {
              return a.offset < b.offset;
            });
  std::uint64_t prev = kHeaderSizeV0;
  for (auto const& span : occupied) {
    if (span.offset < prev) {
      return fail(make_error(FormatErrorCode::OverlappingSpan, span.offset,
                             "span", "placed spans overlap"));
    }
    if (span.offset > prev) {
      pads.push_back(PadRegion{prev, span.offset - prev});
    }
    auto end = checked_add(span.offset, span.length, span.offset, "span");
    if (!end) {
      return fail(end.error());
    }
    prev = *end;
  }
  if (impl_->manifest_offset < prev) {
    return fail(make_error(FormatErrorCode::OverlappingSpan,
                           impl_->manifest_offset, "manifest",
                           "manifest overlaps a payload span"));
  }
  if (impl_->manifest_offset > prev) {
    pads.push_back(
        PadRegion{prev, impl_->manifest_offset - prev});
  }
  for (auto const& pad : pads) {
    if (auto st = write_zeros(*impl_->filesystem, impl_->fd.get(), pad.offset,
                              pad.length,
                              "padding");
        !st) {
      return fail(st.error());
    }
  }

  if (auto st = write_at(*impl_->filesystem, impl_->fd.get(),
                         impl_->manifest_offset, manifest,
                         "manifest");
      !st) {
    return fail(st.error());
  }

  std::array<std::byte, kHeaderSizeV0> header_bytes{};
  if (auto st = encode(header, header_bytes); !st) {
    return fail(st.error());
  }
  if (auto st = write_at(*impl_->filesystem, impl_->fd.get(), 0, header_bytes,
                         "header"); !st) {
    return fail(st.error());
  }

  if (impl_->filesystem->fsync(impl_->filesystem->context, impl_->fd.get()) != 0) {
    return fail(io_error(0, "destination", std::strerror(errno)));
  }
  if (impl_->fd.close() != 0) {
    return fail(io_error(0, "destination", std::strerror(errno)));
  }

  if (impl_->filesystem->rename(impl_->filesystem->context,
                                impl_->temp_path.c_str(),
                                impl_->destination.c_str()) != 0) {
    int const saved_errno = errno;
    int const cleanup_result =
        impl_->filesystem->remove(impl_->filesystem->context,
                                  impl_->temp_path.c_str());
    int const cleanup_errno = errno;
    impl_->closed = true;
    std::string detail = std::strerror(saved_errno);
    if (cleanup_result != 0) {
      detail += "; temporary cleanup failed: ";
      detail += std::strerror(cleanup_errno);
    }
    return std::unexpected(make_error(FormatErrorCode::PublishFailed, 0,
                                      "destination", detail));
  }
  impl_->published = true;
  impl_->closed = true;

  auto const parent = impl_->destination.parent_path().empty()
                          ? std::filesystem::path(".")
                          : impl_->destination.parent_path();
  int const dirfd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dirfd >= 0) {
    UniqueFd dir{dirfd};
    ::fsync(dir.get());
  }

  ArtifactIdentity id{};
  id.path = impl_->destination;
  id.manifest_digest = digest;
  id.size_bytes = impl_->manifest_offset + static_cast<std::uint64_t>(*encoded_n);
  id.manifest_offset = impl_->manifest_offset;
  id.manifest_length = static_cast<std::uint64_t>(*encoded_n);
  id.compiler = impl_->schema.compiler;
  return id;
} catch (std::bad_alloc const&) {
  if (impl_ && !impl_->published) {
    impl_->abandon();
  }
  return std::unexpected(FormatError{.code = FormatErrorCode::AllocationFailure,
                                     .offset = 0,
                                     .field = {},
                                     .detail = {}});
} catch (std::length_error const&) {
  if (impl_ && !impl_->published) {
    impl_->abandon();
  }
  return std::unexpected(make_error(
      FormatErrorCode::ResourceLimitExceeded, 0, "writer",
      "writer state size is not representable while finalizing artifact"));
}

ArtifactBuilder& ArtifactBuilder::destination(std::filesystem::path path) {
  destination_ = std::move(path);
  has_destination_ = true;
  return *this;
}

ArtifactBuilder& ArtifactBuilder::schema(ArtifactSchema schema) {
  schema_ = std::move(schema);
  has_schema_ = true;
  return *this;
}

std::expected<ArtifactWriter, FormatError> ArtifactBuilder::open() const {
  if (!has_destination_ || !has_schema_) {
    return std::unexpected(make_error(
        FormatErrorCode::InconsistentInput, 0, "builder",
        "builder requires destination and schema before open"));
  }
  return ArtifactWriter::create(destination_, schema_);
}

}  // namespace qw38::format
