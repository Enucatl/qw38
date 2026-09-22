#include "format/format.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using qw38::format::ArtifactBuilder;
using qw38::format::ArtifactSchema;
using qw38::format::ArtifactWriter;
using qw38::format::ByteSpan;
using qw38::format::decode_header;
using qw38::format::decode_schema;
using qw38::format::error_message;
using qw38::format::FormatErrorCode;
using qw38::format::Hash256;
using qw38::format::IntegrityKind;
using qw38::format::kHeaderSizeV0;
using qw38::format::kIntegrityRecordBytes;
using qw38::format::kSpanAlignment;
using qw38::format::LogicalPhysicalMapping;
using qw38::format::LogicalQuantizerId;
using qw38::format::MappingKind;
using qw38::format::PhysicalLayoutId;
using qw38::format::sha256;
using qw38::format::SharedBinding;
using qw38::format::SpanKind;
using qw38::format::StorageClass;
using qw38::format::TensorRecord;
using qw38::format::TensorShape;
using qw38::format::WriterFilesystem;
using qw38::format::v0_precision_policy;

std::atomic<bool> g_fail_next_allocation{false};

void* operator new(std::size_t bytes) {
  if (g_fail_next_allocation.exchange(false, std::memory_order_relaxed)) {
    throw std::bad_alloc();
  }
  if (void* p = std::malloc(bytes == 0 ? 1 : bytes)) {
    return p;
  }
  throw std::bad_alloc();
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

int g_failures = 0;

void fail_next_host_allocation() noexcept {
  g_fail_next_allocation.store(true, std::memory_order_relaxed);
}

void fail(std::string_view what) {
  std::cerr << "FAIL: " << what << '\n';
  ++g_failures;
}

void expect(bool cond, std::string_view what) {
  if (!cond) {
    fail(what);
  }
}

class ScratchDir {
 public:
  ScratchDir() {
    auto base = std::filesystem::temp_directory_path() / "qw38-writer-XXXXXX";
    std::string tmpl = base.string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (::mkdtemp(buf.data()) == nullptr) {
      fail("mkdtemp");
      return;
    }
    path_ = buf.data();
  }

  ~ScratchDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  ScratchDir(ScratchDir const&) = delete;
  ScratchDir& operator=(ScratchDir const&) = delete;

  std::filesystem::path const& path() const { return path_; }
  std::filesystem::path file(std::string_view name) const {
    return path_ / std::string(name);
  }

 private:
  std::filesystem::path path_;
};

std::vector<std::byte> read_all(std::filesystem::path const& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    fail("open " + path.string());
    return {};
  }
  in.seekg(0, std::ios::end);
  auto const n = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<std::byte> bytes(n);
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(n));
  if (!in) {
    fail("read " + path.string());
    return {};
  }
  return bytes;
}

std::size_t count_temp_files(ScratchDir const& dir,
                             std::filesystem::path const& destination) {
  std::size_t count = 0;
  auto const prefix = destination.filename().string() + ".tmp.";
  std::error_code ec;
  for (std::filesystem::directory_iterator it(dir.path(), ec), end; it != end;
       it.increment(ec)) {
    if (ec) {
      fail("iterate temporary files");
      break;
    }
    if (it->path().filename().string().starts_with(prefix)) {
      ++count;
    }
  }
  return count;
}

ArtifactSchema base_schema() {
  ArtifactSchema schema{};
  schema.compiler = {.ident = "qw38", .major = 0, .minor = 1, .patch = 0};
  schema.precision = v0_precision_policy();
  auto const state = qw38::format::v0_language_state_schema();
  schema.state.assign(state.begin(), state.end());
  auto const scratch = qw38::format::v0_language_scratch_schema();
  schema.scratch.assign(scratch.begin(), scratch.end());
  return schema;
}

TensorRecord unplaced_bf16_vector(std::uint32_t id, std::string name,
                                  std::uint64_t n) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape.rank = 1;
  t.shape.logical[0] = n;
  t.shape.padded[0] = n;
  t.storage = StorageClass::Bf16;
  t.quantizer = LogicalQuantizerId::None;
  t.layout = PhysicalLayoutId::CudaBf16VectorV0;
  t.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};
  return t;
}

void test_golden_alignment_and_offsets() {
  ScratchDir dir;
  auto dest = dir.file("golden.qw38");
  auto schema = base_schema();
  schema.tensors.push_back(unplaced_bf16_vector(1, "v", 4));
  auto writer = ArtifactWriter::create(dest, schema);
  if (!writer) {
    fail(std::string("create golden: ") + error_message(writer.error()));
    return;
  }
  std::array<std::byte, 8> payload{
      std::byte{0x00}, std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
      std::byte{0x44}, std::byte{0x55}, std::byte{0x66}, std::byte{0x77}};
  auto wr = writer->write_span("v", SpanKind::Payload, payload);
  if (!wr) {
    fail(std::string("write golden: ") + error_message(wr.error()));
    return;
  }
  auto id = writer->finalize();
  if (!id) {
    fail(std::string("finalize golden: ") + error_message(id.error()));
    return;
  }
  expect(id->path == dest, "identity path is destination");
  expect(id->manifest_offset == 512, "manifest starts at 512 after 8-byte span");
  expect(id->compiler.ident == "qw38", "identity carries compiler revision");

  auto bytes = read_all(dest);
  expect(bytes.size() == id->size_bytes, "identity size matches file");
  expect(bytes.size() >= kHeaderSizeV0, "file contains a header");
  expect(static_cast<char>(bytes[0]) == 'Q' &&
             static_cast<char>(bytes[1]) == 'W' &&
             static_cast<char>(bytes[2]) == '3' &&
             static_cast<char>(bytes[3]) == '8' &&
             static_cast<char>(bytes[4]) == 'F' &&
             static_cast<char>(bytes[5]) == 'M' &&
             static_cast<char>(bytes[6]) == 'T' &&
             bytes[7] == std::byte{0},
         "header magic QW38FMT");
  expect(static_cast<std::uint8_t>(bytes[8]) == 1 &&
             static_cast<std::uint8_t>(bytes[9]) == 0,
         "container version 1 LE");
  expect(static_cast<std::uint8_t>(bytes[16]) == 0x00 &&
             static_cast<std::uint8_t>(bytes[17]) == 0x02,
         "manifest offset 512 LE");
  for (std::size_t i = 64; i < 256; ++i) {
    expect(bytes[i] == std::byte{0}, "header-to-payload padding is zero");
  }
  for (std::size_t i = 0; i < payload.size(); ++i) {
    expect(bytes[256 + i] == payload[i], "payload bytes at offset 256");
  }
  for (std::size_t i = 264; i < 512; ++i) {
    expect(bytes[i] == std::byte{0}, "payload-to-manifest padding is zero");
  }

  auto header = decode_header(std::span<std::byte const>{bytes.data(), 64});
  if (!header) {
    fail(std::string("decode header: ") + error_message(header.error()));
    return;
  }
  expect(header->manifest_offset == 512, "decoded manifest offset");
  auto manifest = decode_schema(std::span<std::byte const>{
      bytes.data() + header->manifest_offset,
      static_cast<std::size_t>(header->manifest_length)});
  if (!manifest) {
    fail(std::string("decode golden schema: ") + error_message(manifest.error()));
    return;
  }
  expect(manifest->tensors.size() == 1, "one tensor");
  expect(manifest->tensors[0].payload.offset == 256, "payload base 256");
  expect(manifest->tensors[0].payload.length == 8, "payload length 8");
  expect(manifest->tensors[0].payload.offset % kSpanAlignment == 0,
         "payload 256-aligned");
  expect(!manifest->integrity.empty() &&
             manifest->integrity.back().kind == IntegrityKind::Sha256Manifest,
         "Sha256Manifest is last integrity record");
  auto const& mrec = manifest->integrity.back();
  expect(mrec.region.offset == header->manifest_offset, "manifest hash region");
  expect(mrec.region.length == header->manifest_length,
         "manifest hash covers the complete manifest");
  std::vector<std::byte> canonical(bytes.begin() +
                                       static_cast<std::ptrdiff_t>(mrec.region.offset),
                                   bytes.begin() + static_cast<std::ptrdiff_t>(
                                                       mrec.region.offset +
                                                       mrec.region.length));
  std::fill(canonical.end() - qw38::format::kHashBytes, canonical.end(),
            std::byte{});
  Hash256 const manifest_digest = sha256(canonical);
  expect(manifest_digest == mrec.digest,
         "independent SHA-256 of manifest with self-digest zeroed");
  expect(manifest_digest == id->manifest_digest,
         "identity digest matches record");
}

void test_deterministic_repeat() {
  ScratchDir dir;
  auto schema = base_schema();
  schema.tensors.push_back(unplaced_bf16_vector(1, "v", 4));
  std::array<std::byte, 8> payload{
      std::byte{0xA0}, std::byte{0xA1}, std::byte{0xA2}, std::byte{0xA3},
      std::byte{0xA4}, std::byte{0xA5}, std::byte{0xA6}, std::byte{0xA7}};
  auto write_one = [&](std::string_view name) {
    auto dest = dir.file(std::string(name));
    ArtifactBuilder b;
    b.destination(dest).schema(schema);
    auto writer = b.open();
    expect(static_cast<bool>(writer), "builder open");
    if (!writer) {
      return std::vector<std::byte>{};
    }
    expect(static_cast<bool>(writer->write_span("v", SpanKind::Payload, payload)),
           "write payload");
    auto id = writer->finalize();
    expect(static_cast<bool>(id), "finalize repeat");
    return read_all(dest);
  };
  auto a = write_one("a.qw38");
  auto b = write_one("b.qw38");
  expect(a == b && !a.empty(), "same inputs yield byte-identical artifacts");
}

void test_empty_duplicate_missing() {
  ScratchDir dir;
  auto schema = base_schema();
  schema.tensors.push_back(unplaced_bf16_vector(1, "v", 4));
  auto dest = dir.file("empty.qw38");
  auto writer = ArtifactWriter::create(dest, schema);
  if (!writer) {
    fail("create empty case");
    return;
  }
  auto empty = writer->write_span("v", SpanKind::Payload, {});
  expect(!empty && empty.error().code == FormatErrorCode::EmptySpan,
         "empty chunk is rejected");

  std::array<std::byte, 8> payload{};
  payload.fill(std::byte{1});
  expect(static_cast<bool>(
             writer->write_span("v", SpanKind::Payload, payload)),
         "full payload write");
  auto dup = writer->write_span("v", SpanKind::Payload, payload);
  expect(!dup && dup.error().code == FormatErrorCode::DuplicateSpan,
         "duplicate span is rejected");
  auto id = writer->finalize();
  expect(static_cast<bool>(id), "duplicate does not poison a complete artifact");

  auto dest2 = dir.file("missing.qw38");
  auto writer2 = ArtifactWriter::create(dest2, schema);
  if (!writer2) {
    fail("create missing case");
    return;
  }
  auto missing = writer2->finalize();
  expect(!missing && missing.error().code == FormatErrorCode::MissingSpan,
         "missing span is rejected at finalize");
  expect(!std::filesystem::exists(dest2),
         "failed finalize does not publish destination");
  expect(count_temp_files(dir, dest2) == 0,
         "failed finalize removes the owned temp file");

  auto dest3 = dir.file("incomplete.qw38");
  auto writer3 = ArtifactWriter::create(dest3, schema);
  if (!writer3) {
    fail("create incomplete case");
    return;
  }
  std::array<std::byte, 3> partial{std::byte{1}, std::byte{2}, std::byte{3}};
  expect(static_cast<bool>(
             writer3->write_span("v", SpanKind::Payload, partial)),
         "partial write");
  auto incomplete = writer3->finalize();
  expect(!incomplete &&
             incomplete.error().code == FormatErrorCode::IncompleteSpan,
         "partial span is rejected at finalize");
  expect(!std::filesystem::exists(dest3),
         "incomplete finalize does not publish destination");
}

void test_overflow_and_inconsistent() {
  auto schema = base_schema();
  auto huge = unplaced_bf16_vector(1, "huge", 4);
  huge.shape.logical[0] = (std::numeric_limits<std::uint64_t>::max() / 2) + 1;
  huge.shape.padded[0] = huge.shape.logical[0];
  schema.tensors.push_back(huge);
  auto overflowed = ArtifactWriter::create("/tmp/unused.qw38", schema);
  expect(!overflowed && overflowed.error().code == FormatErrorCode::Overflow,
         "payload size overflow is rejected before publication");

  schema = base_schema();
  auto t = unplaced_bf16_vector(1, "v", 4);
  t.payload = ByteSpan{.offset = 128, .length = 8};
  schema.tensors.push_back(t);
  auto misaligned = ArtifactWriter::create("/tmp/unused.qw38", schema);
  expect(!misaligned && misaligned.error().code == FormatErrorCode::Misaligned,
         "misaligned input span is rejected");

  schema = base_schema();
  auto a = unplaced_bf16_vector(1, "a", 4);
  auto b = unplaced_bf16_vector(2, "b", 4);
  a.payload = ByteSpan{.offset = 256, .length = 8};
  b.payload = ByteSpan{.offset = 256, .length = 8};
  schema.tensors.push_back(a);
  schema.tensors.push_back(b);
  auto overlap = ArtifactWriter::create("/tmp/unused.qw38", schema);
  expect(!overlap && overlap.error().code == FormatErrorCode::OverlappingSpan,
         "overlapping input spans are rejected");

  schema = base_schema();
  t = unplaced_bf16_vector(1, "v", 4);
  t.payload.length = 3;
  schema.tensors.push_back(t);
  auto inconsistent = ArtifactWriter::create("/tmp/unused.qw38", schema);
  expect(!inconsistent &&
             inconsistent.error().code == FormatErrorCode::InconsistentInput,
         "payload length mismatch is rejected");
}

void test_invalid_input_span_placements() {
  auto rejects = [](ArtifactSchema schema, FormatErrorCode code,
                    std::string_view what) {
    auto writer = ArtifactWriter::create("/tmp/unused.qw38", schema);
    expect(!writer && writer.error().code == code, what);
  };

  auto schema = base_schema();
  auto t = unplaced_bf16_vector(1, "v", 4);
  t.payload = ByteSpan{.offset = 256, .length = 0};
  schema.tensors = {t};
  rejects(schema, FormatErrorCode::InvalidSpan,
          "empty payload with a nonzero offset is rejected");

  t.payload = ByteSpan{};
  t.scales = ByteSpan{.offset = 256, .length = 0};
  schema.tensors = {t};
  rejects(schema, FormatErrorCode::InvalidSpan,
          "empty scale span with a nonzero offset is rejected");

  t.payload = ByteSpan{.offset = 0, .length = 8};
  t.scales = ByteSpan{};
  schema.tensors = {t};
  rejects(schema, FormatErrorCode::InvalidSpan,
          "nonempty payload without a placement offset is rejected");

  t = unplaced_bf16_vector(1, "v", 256);
  t.payload = ByteSpan{.offset = std::numeric_limits<std::uint64_t>::max() -
                                   255,
                       .length = 512};
  schema.tensors = {t};
  rejects(schema, FormatErrorCode::Overflow,
          "overflowing payload placement is rejected");

  schema = base_schema();
  auto owner = unplaced_bf16_vector(1, "owner", 4);
  auto alias = unplaced_bf16_vector(2, "alias", 4);
  owner.payload = ByteSpan{.offset = 256, .length = 8};
  alias.payload = ByteSpan{.offset = 512, .length = 8};
  schema.tensors = {owner, alias};
  schema.shared_bindings = {
      SharedBinding{.owner_tensor_id = 1, .alias_tensor_id = 2},
  };
  alias.payload = ByteSpan{.offset = 128, .length = 8};
  schema.tensors = {owner, alias};
  rejects(schema, FormatErrorCode::Misaligned,
          "misaligned alias payload placement is rejected");

  alias.payload = ByteSpan{.offset = 512, .length = 8};
  schema.tensors = {owner, alias};
  rejects(schema, FormatErrorCode::SharedBinding,
          "alias payload placement must exactly match its owner");

  alias.payload = owner.payload;
  schema.tensors = {owner, alias};
  auto writer = ArtifactWriter::create("/tmp/unused.qw38", schema);
  expect(static_cast<bool>(writer),
         "identical owner and alias input placements are permitted");
}

void test_invalid_shared_ownership_rejected_before_writing() {
  auto schema = base_schema();
  schema.tensors = {
      unplaced_bf16_vector(1, "a", 4),
      unplaced_bf16_vector(2, "b", 4),
      unplaced_bf16_vector(3, "c", 4),
  };
  schema.shared_bindings = {
      SharedBinding{.owner_tensor_id = 1, .alias_tensor_id = 2},
      SharedBinding{.owner_tensor_id = 3, .alias_tensor_id = 2},
  };
  auto writer = ArtifactWriter::create("/tmp/unused.qw38", schema);
  expect(!writer && writer.error().code == FormatErrorCode::SharedBinding,
         "writer rejects multiple owners before creating a file");

  schema.shared_bindings = {
      SharedBinding{.owner_tensor_id = 1, .alias_tensor_id = 2},
      SharedBinding{.owner_tensor_id = 2, .alias_tensor_id = 3},
  };
  writer = ArtifactWriter::create("/tmp/unused.qw38", schema);
  expect(!writer && writer.error().code == FormatErrorCode::SharedBinding,
         "writer rejects alias ownership chains before creating a file");
}

void test_failure_cleanup() {
  ScratchDir dir;
  auto dest = dir.file("cleanup.qw38");
  auto preexisting = dir.file("keep.qw38");
  {
    std::ofstream out(preexisting, std::ios::binary);
    out << "OLD";
  }
  auto schema = base_schema();
  schema.tensors.push_back(unplaced_bf16_vector(1, "v", 4));
  {
    auto writer = ArtifactWriter::create(dest, schema);
    expect(static_cast<bool>(writer), "create for destructor cleanup");
    expect(count_temp_files(dir, dest) == 1,
           "unique temp file exists while writer is open");
  }
  expect(!std::filesystem::exists(dest),
         "destroying an unfinalized writer leaves no destination");
  expect(count_temp_files(dir, dest) == 0,
         "destroying an unfinalized writer removes the owned temp file");

  auto writer = ArtifactWriter::create(preexisting, schema);
  expect(static_cast<bool>(writer), "create over existing dest");
  auto missing = writer->finalize();
  expect(!missing, "finalize fails without spans");
  expect(std::filesystem::exists(preexisting), "preexisting dest remains");
  auto kept = read_all(preexisting);
  expect(kept.size() == 3 && static_cast<char>(kept[0]) == 'O',
         "failed finalize does not replace an existing destination");

  auto destination_with_temp = dir.file("preexisting-temp.qw38");
  auto preexisting_temp = destination_with_temp.string() + ".tmp";
  {
    std::ofstream out(preexisting_temp, std::ios::binary);
    out << "KEEP";
  }
  auto writer_with_temp =
      ArtifactWriter::create(destination_with_temp, schema);
  expect(static_cast<bool>(writer_with_temp),
         "create with pre-existing temporary file");
  expect(std::filesystem::exists(preexisting_temp),
         "pre-existing temporary file is retained at create");
  expect(count_temp_files(dir, destination_with_temp) == 1,
         "writer owns one unique temporary file");
  if (writer_with_temp) {
    auto failed = writer_with_temp->finalize();
    expect(!failed, "finalize with pre-existing temporary file fails missing");
  }
  expect(std::filesystem::exists(preexisting_temp),
         "cleanup retains unrelated pre-existing temporary file");
  expect(read_all(preexisting_temp).size() == 4,
         "pre-existing temporary file contents are unchanged");
  expect(count_temp_files(dir, destination_with_temp) == 0,
         "owned temporary file is removed after failure");
}

void test_writer_interleaving_owns_publication() {
  ScratchDir dir;
  auto dest = dir.file("concurrent.qw38");
  auto schema = base_schema();
  schema.tensors.push_back(unplaced_bf16_vector(1, "v", 4));

  auto writer_a = ArtifactWriter::create(dest, schema);
  auto writer_b = ArtifactWriter::create(dest, schema);
  expect(static_cast<bool>(writer_a) && static_cast<bool>(writer_b),
         "two writers can open the same destination");
  if (!writer_a || !writer_b) {
    return;
  }
  expect(count_temp_files(dir, dest) == 2,
         "concurrent writers receive distinct temporary files");

  std::array<std::byte, 8> payload_a{};
  payload_a.fill(std::byte{0xA1});
  std::array<std::byte, 8> payload_b{};
  payload_b.fill(std::byte{0xB2});
  expect(static_cast<bool>(
             writer_a->write_span("v", SpanKind::Payload, payload_a)),
         "writer A writes its payload");

  auto id_a = writer_a->finalize();
  expect(static_cast<bool>(id_a), "writer A publishes its own complete file");
  if (id_a) {
    auto bytes = read_all(dest);
    expect(bytes.size() == id_a->size_bytes,
           "writer A publication has the complete artifact size");
    expect(bytes.size() > 256 && bytes[256] == std::byte{0xA1},
           "writer A publication contains writer A payload");
  }

  expect(static_cast<bool>(
             writer_b->write_span("v", SpanKind::Payload, payload_b)),
         "writer B writes its payload");
  auto id_b = writer_b->finalize();
  expect(static_cast<bool>(id_b), "writer B publishes its own complete file");
  if (id_b) {
    auto bytes = read_all(dest);
    expect(bytes.size() == id_b->size_bytes,
           "final destination has the complete writer B artifact");
    expect(bytes.size() > 256 && bytes[256] == std::byte{0xB2},
           "final destination contains writer B payload");
  }
  expect(count_temp_files(dir, dest) == 0,
         "all owned temporary files are cleaned after publication");
}

void test_chunked_stream() {
  ScratchDir dir;
  auto dest = dir.file("chunks.qw38");
  auto schema = base_schema();
  schema.tensors.push_back(unplaced_bf16_vector(1, "v", 4));
  auto writer = ArtifactWriter::create(dest, schema);
  if (!writer) {
    fail("create chunked");
    return;
  }
  std::array<std::byte, 4> a{std::byte{9}, std::byte{8}, std::byte{7},
                             std::byte{6}};
  std::array<std::byte, 4> b{std::byte{5}, std::byte{4}, std::byte{3},
                             std::byte{2}};
  expect(static_cast<bool>(writer->write_span("v", SpanKind::Payload, a)),
         "first chunk");
  expect(static_cast<bool>(writer->write_span("v", SpanKind::Payload, b)),
         "second chunk");
  auto id = writer->finalize();
  expect(static_cast<bool>(id), "finalize chunked");
  auto bytes = read_all(dest);
  expect(bytes.size() > 256 + 8, "chunked file size");
  expect(bytes[256] == std::byte{9} && bytes[259] == std::byte{6} &&
             bytes[260] == std::byte{5} && bytes[263] == std::byte{2},
         "chunked payload concatenated in order");
}

enum class FaultStage {
  PayloadWrite,
  ManifestWrite,
  Fsync,
  Close,
  Rename,
  RenameAndCleanup,
};

struct FaultFilesystem {
  FaultStage stage;

  static ssize_t pwrite(void* context, int fd, void const* data,
                        std::size_t size, off_t offset) {
    auto const stage = static_cast<FaultFilesystem*>(context)->stage;
    if ((stage == FaultStage::PayloadWrite && offset == 256) ||
        (stage == FaultStage::ManifestWrite && offset >= 512)) {
      errno = EIO;
      return -1;
    }
    return ::pwrite(fd, data, size, offset);
  }

  static int fsync(void* context, int fd) {
    if (static_cast<FaultFilesystem*>(context)->stage == FaultStage::Fsync) {
      errno = EIO;
      return -1;
    }
    return ::fsync(fd);
  }

  static int close(void* context, int fd) {
    int const result = ::close(fd);
    if (static_cast<FaultFilesystem*>(context)->stage == FaultStage::Close) {
      errno = EIO;
      return -1;
    }
    return result;
  }

  static int rename(void* context, char const* from, char const* to) {
    auto const stage = static_cast<FaultFilesystem*>(context)->stage;
    if (stage == FaultStage::Rename || stage == FaultStage::RenameAndCleanup) {
      errno = EIO;
      return -1;
    }
    return ::rename(from, to);
  }

  static int remove(void* context, char const* path) {
    if (static_cast<FaultFilesystem*>(context)->stage ==
        FaultStage::RenameAndCleanup) {
      errno = EACCES;
      return -1;
    }
    return ::unlink(path);
  }

  [[nodiscard]] WriterFilesystem operations() {
    return WriterFilesystem{.pwrite = pwrite,
                            .fsync = fsync,
                            .close = close,
                            .rename = rename,
                            .remove = remove,
                            .context = this};
  }
};

void test_publication_io_failure_cleanup() {
  struct Case {
    FaultStage stage;
    std::string_view name;
    FormatErrorCode error;
  };
  std::array const cases{
      Case{FaultStage::PayloadWrite, "payload-pwrite", FormatErrorCode::IoFailure},
      Case{FaultStage::ManifestWrite, "manifest-pwrite", FormatErrorCode::IoFailure},
      Case{FaultStage::Fsync, "fsync", FormatErrorCode::IoFailure},
      Case{FaultStage::Close, "close", FormatErrorCode::IoFailure},
      Case{FaultStage::Rename, "rename", FormatErrorCode::PublishFailed},
      Case{FaultStage::RenameAndCleanup, "rename-cleanup",
           FormatErrorCode::PublishFailed},
  };
  for (auto const& test : cases) {
    ScratchDir dir;
    auto const dest = dir.file(std::string(test.name) + ".qw38");
    {
      std::ofstream out(dest, std::ios::binary);
      out << "OLD";
    }
    auto schema = base_schema();
    schema.tensors.push_back(unplaced_bf16_vector(1, "v", 4));
    FaultFilesystem fault{test.stage};
    auto operations = fault.operations();
    {
      auto writer = ArtifactWriter::create(dest, schema, &operations);
      expect(static_cast<bool>(writer), "fault-injected writer opens");
      if (writer) {
        std::array<std::byte, 8> payload{};
        auto write = writer->write_span("v", SpanKind::Payload, payload);
        if (test.stage == FaultStage::PayloadWrite) {
          expect(!write && write.error().code == test.error,
                 "injected payload pwrite fails");
        } else {
          expect(static_cast<bool>(write), "payload succeeds before publication fault");
          auto final = writer->finalize();
          expect(!final && final.error().code == test.error,
                 "injected publication stage fails");
          if (test.stage == FaultStage::RenameAndCleanup && !final) {
            expect(final.error().detail.find("Input/output error") !=
                       std::string::npos,
                   "rename error remains the primary failure");
            expect(final.error().detail.find("temporary cleanup failed") !=
                       std::string::npos,
                   "cleanup failure is retained as secondary context");
            fault.stage = FaultStage::Rename;
          }
        }
      }
    }
    expect(read_all(dest) == std::vector<std::byte>{std::byte{'O'}, std::byte{'L'}, std::byte{'D'}},
           "publication failure preserves destination");
    expect(count_temp_files(dir, dest) == 0,
           "publication failure removes owned temporary file");
  }
}

void test_writer_allocation_failure_translation() {
  ScratchDir dir;
  auto schema = base_schema();
  std::string const name(64, 'v');
  schema.tensors.push_back(unplaced_bf16_vector(1, name, 4));
  auto create_destination = dir.file("create.qw38");

  fail_next_host_allocation();
  auto create = ArtifactWriter::create(create_destination, schema);
  expect(!create && create.error().code == FormatErrorCode::AllocationFailure,
         "create allocation failure is typed");

  auto writer = ArtifactWriter::create(dir.file("write.qw38"), schema);
  expect(static_cast<bool>(writer), "writer opens before write allocation test");
  if (writer) {
    std::array<std::byte, 8> payload{};
    fail_next_host_allocation();
    auto write = writer->write_span(name, SpanKind::Payload, payload);
    expect(!write && write.error().code == FormatErrorCode::AllocationFailure,
           "write allocation failure is typed");
    expect(count_temp_files(dir, dir.file("write.qw38")) == 0,
           "write allocation failure cleans up temporary file");
  }

  auto final_writer = ArtifactWriter::create(dir.file("finalize.qw38"), schema);
  expect(static_cast<bool>(final_writer),
         "writer opens before finalize allocation test");
  if (final_writer) {
    std::array<std::byte, 8> payload{};
    expect(static_cast<bool>(final_writer->write_span(name, SpanKind::Payload,
                                                       payload)),
           "write succeeds before finalize allocation test");
    fail_next_host_allocation();
    auto final = final_writer->finalize();
    expect(!final && final.error().code == FormatErrorCode::AllocationFailure,
           "finalize allocation failure is typed");
    expect(count_temp_files(dir, dir.file("finalize.qw38")) == 0,
           "finalize allocation failure cleans up temporary file");
  }
}

}  // namespace

int main() {
  test_golden_alignment_and_offsets();
  test_deterministic_repeat();
  test_empty_duplicate_missing();
  test_overflow_and_inconsistent();
  test_invalid_input_span_placements();
  test_invalid_shared_ownership_rejected_before_writing();
  test_failure_cleanup();
  test_writer_interleaving_owns_publication();
  test_chunked_stream();
  test_publication_io_failure_cleanup();
  test_writer_allocation_failure_translation();
  if (g_failures != 0) {
    std::cerr << g_failures << " format writer checks failed\n";
    return 1;
  }
  std::cout << "format writer unit tests ok\n";
  return 0;
}
