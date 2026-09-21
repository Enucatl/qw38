#include "format/writer.hpp"

#include "format/constants.hpp"
#include "format/layout.hpp"
#include "format/sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

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
  [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }

  void reset() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int release() noexcept {
    int const fd = fd_;
    fd_ = -1;
    return fd;
  }

 private:
  int fd_{-1};
};

std::expected<void, FormatError> write_at(int fd, std::uint64_t offset,
                                          std::span<std::byte const> data,
                                          std::string_view field) {
  while (!data.empty()) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      return std::unexpected(make_error(FormatErrorCode::Overflow, offset, field,
                                        "file offset exceeds off_t"));
    }
    ssize_t const n = ::pwrite(fd, data.data(), data.size(),
                               static_cast<off_t>(offset));
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

std::expected<void, FormatError> write_zeros(int fd, std::uint64_t offset,
                                             std::uint64_t length,
                                             std::string_view field) {
  std::array<std::byte, 256> zeros{};
  while (length > 0) {
    std::size_t const n =
        static_cast<std::size_t>(std::min<std::uint64_t>(length, zeros.size()));
    if (auto st = write_at(fd, offset,
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

std::uint32_t owner_of(ArtifactSchema const& schema, std::uint32_t tensor_id) {
  for (auto const& b : schema.shared_bindings) {
    if (b.alias_tensor_id == tensor_id) {
      return b.owner_tensor_id;
    }
  }
  return tensor_id;
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
    auto const* owner = find_tensor(schema, owner_id);
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

std::expected<ArtifactSchema, FormatError> prepare_schema(
    ArtifactSchema const& input) {
  if (!input.integrity.empty()) {
    return std::unexpected(make_error(
        FormatErrorCode::InconsistentInput, 0, "integrity",
        "writer owns integrity records; input must leave them empty"));
  }
  ArtifactSchema schema = input;
  std::vector<ByteSpan> input_unique_spans;
  for (auto const& tensor : schema.tensors) {
    auto const payload_n = expected_payload_bytes(tensor);
    if (!payload_n) {
      return std::unexpected(payload_n.error());
    }
    auto const scale_n = expected_scale_bytes(tensor);
    if (!scale_n) {
      return std::unexpected(scale_n.error());
    }
    if (tensor.payload.length != 0 && tensor.payload.length != *payload_n) {
      return std::unexpected(make_error(
          FormatErrorCode::InconsistentInput, tensor.payload.offset,
          "tensor.payload.length",
          "declared payload length does not match layout"));
    }
    if (tensor.scales.length != 0 && tensor.scales.length != *scale_n) {
      return std::unexpected(make_error(
          FormatErrorCode::InconsistentInput, tensor.scales.offset,
          "tensor.scales.length",
          "declared scale length does not match quantizer"));
    }
    if (is_alias(schema, tensor.tensor_id)) {
      continue;
    }
    if (tensor.payload.length != 0) {
      if (auto st = require_span_alignment(tensor.payload.offset,
                                           tensor.payload.length,
                                           "tensor.payload");
          !st) {
        return std::unexpected(st.error());
      }
      if (auto st = checked_add(tensor.payload.offset, tensor.payload.length,
                                tensor.payload.offset, "tensor.payload");
          !st) {
        return std::unexpected(st.error());
      }
      input_unique_spans.push_back(tensor.payload);
    }
    if (tensor.scales.length != 0) {
      if (auto st = require_span_alignment(tensor.scales.offset,
                                           tensor.scales.length,
                                           "tensor.scales");
          !st) {
        return std::unexpected(st.error());
      }
      if (auto st = checked_add(tensor.scales.offset, tensor.scales.length,
                                tensor.scales.offset, "tensor.scales");
          !st) {
        return std::unexpected(st.error());
      }
      input_unique_spans.push_back(tensor.scales);
    }
  }
  for (std::size_t i = 0; i < input_unique_spans.size(); ++i) {
    for (std::size_t j = i + 1; j < input_unique_spans.size(); ++j) {
      if (spans_overlap(input_unique_spans[i], input_unique_spans[j])) {
        return std::unexpected(make_error(
            FormatErrorCode::OverlappingSpan, input_unique_spans[i].offset,
            "tensor.payload", "input payload/scale spans overlap"));
      }
    }
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
  std::uint64_t manifest_offset{};
  bool published{false};
  bool closed{false};

  struct PhysicalSpan {
    std::uint32_t owner_tensor_id{};
    SpanKind kind{SpanKind::Payload};
    ByteSpan region{};
    std::string field;
    Sha256 hasher{};
    std::uint64_t written{};
    bool started{false};
    bool complete{false};
  };

  std::vector<PhysicalSpan> spans;
  std::unordered_map<std::string, std::size_t> payload_by_name;
  std::unordered_map<std::string, std::size_t> scales_by_name;

  void abandon() {
    fd.reset();
    if (!published && !temp_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(temp_path, ec);
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
    std::filesystem::path destination, ArtifactSchema const& schema) {
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
  impl->destination = std::move(destination);
  impl->schema = std::move(*prepared);
  impl->manifest_offset = *manifest_off;
  impl->temp_path = impl->destination;
  impl->temp_path += ".tmp";

  std::error_code ec;
  auto const parent = impl->destination.parent_path();
  if (!parent.empty() && !std::filesystem::is_directory(parent, ec)) {
    return std::unexpected(io_error(0, "destination",
                                    "parent directory does not exist"));
  }
  std::filesystem::remove(impl->temp_path, ec);

  int const raw = ::open(impl->temp_path.c_str(),
                         O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0644);
  if (raw < 0) {
    return std::unexpected(io_error(0, "destination", std::strerror(errno)));
  }
  impl->fd = UniqueFd{raw};

  if (auto st = write_zeros(impl->fd.get(), 0, kHeaderSizeV0, "header"); !st) {
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
    auto const payload_it = payload_index.find(owner_id);
    if (payload_it != payload_index.end()) {
      impl->payload_by_name.emplace(tensor.logical_name, payload_it->second);
    }
    auto const scale_it = scale_index.find(owner_id);
    if (scale_it != scale_index.end()) {
      impl->scales_by_name.emplace(tensor.logical_name, scale_it->second);
    }
  }

  return ArtifactWriter{std::move(impl)};
}

std::expected<void, FormatError> ArtifactWriter::write_span(
    std::string_view logical_name, SpanKind kind,
    std::span<std::byte const> chunk) {
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
  if (auto st = write_at(impl_->fd.get(), offset, chunk, span.field); !st) {
    return st;
  }
  span.hasher.update(chunk);
  span.written += chunk.size();
  span.started = true;
  if (span.written == span.region.length) {
    span.complete = true;
  }
  return {};
}

std::expected<ArtifactIdentity, FormatError> ArtifactWriter::finalize() {
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

  std::vector<IntegrityRecord> records;
  records.reserve(impl_->schema.tensors.size() * 2 + 1);
  for (auto const& tensor : impl_->schema.tensors) {
    auto const owner_id = owner_of(impl_->schema, tensor.tensor_id);
    for (auto const& span : impl_->spans) {
      if (span.owner_tensor_id != owner_id) {
        continue;
      }
      if (span.kind == SpanKind::Payload) {
        IntegrityRecord rec{};
        rec.kind = IntegrityKind::Sha256PayloadSpan;
        rec.tensor_id = tensor.tensor_id;
        rec.region = tensor.payload;
        rec.digest = span.hasher.digest();
        records.push_back(rec);
      } else {
        IntegrityRecord rec{};
        rec.kind = IntegrityKind::Sha256ScaleSpan;
        rec.tensor_id = tensor.tensor_id;
        rec.region = tensor.scales;
        rec.digest = span.hasher.digest();
        records.push_back(rec);
      }
    }
  }

  IntegrityRecord manifest_rec{};
  manifest_rec.kind = IntegrityKind::Sha256Manifest;
  records.push_back(manifest_rec);
  impl_->schema.integrity = records;

  auto encoded_n = encoded_size(impl_->schema);
  if (!encoded_n) {
    return fail(encoded_n.error());
  }
  if (*encoded_n < kIntegrityRecordBytes) {
    return fail(make_error(FormatErrorCode::InvalidSpan, 0, "manifest",
                           "encoded manifest is smaller than one record"));
  }
  std::uint64_t const prefix_len =
      static_cast<std::uint64_t>(*encoded_n) - kIntegrityRecordBytes;
  impl_->schema.integrity.back().region = ByteSpan{
      .offset = impl_->manifest_offset,
      .length = prefix_len,
  };

  std::vector<std::byte> manifest(*encoded_n);
  if (auto st = encode(impl_->schema, manifest); !st) {
    return fail(st.error());
  }
  Hash256 const digest =
      sha256(std::span<std::byte const>{manifest.data(),
                                        static_cast<std::size_t>(prefix_len)});
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
    if (auto st = write_zeros(impl_->fd.get(), pad.offset, pad.length,
                              "padding");
        !st) {
      return fail(st.error());
    }
  }

  if (auto st = write_at(impl_->fd.get(), impl_->manifest_offset, manifest,
                         "manifest");
      !st) {
    return fail(st.error());
  }

  std::array<std::byte, kHeaderSizeV0> header_bytes{};
  if (auto st = encode(header, header_bytes); !st) {
    return fail(st.error());
  }
  if (auto st = write_at(impl_->fd.get(), 0, header_bytes, "header"); !st) {
    return fail(st.error());
  }

  if (::fsync(impl_->fd.get()) != 0) {
    return fail(io_error(0, "destination", std::strerror(errno)));
  }
  impl_->fd.reset();

  std::error_code ec;
  std::filesystem::rename(impl_->temp_path, impl_->destination, ec);
  if (ec) {
    std::filesystem::remove(impl_->temp_path, ec);
    impl_->closed = true;
    return std::unexpected(make_error(FormatErrorCode::PublishFailed, 0,
                                      "destination", ec.message()));
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
