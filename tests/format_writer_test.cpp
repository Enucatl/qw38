#include "format/format.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
using qw38::format::SpanKind;
using qw38::format::StorageClass;
using qw38::format::TensorRecord;
using qw38::format::TensorShape;
using qw38::format::v0_precision_policy;

namespace {

int g_failures = 0;

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
  expect(mrec.region.length == header->manifest_length - kIntegrityRecordBytes,
         "manifest hash excludes the self-digest record");
  Hash256 const prefix = sha256(std::span<std::byte const>{
      bytes.data() + mrec.region.offset,
      static_cast<std::size_t>(mrec.region.length)});
  expect(prefix == mrec.digest, "independent SHA-256 of hashed prefix");
  expect(prefix == id->manifest_digest, "identity digest matches record");
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

}  // namespace

int main() {
  test_golden_alignment_and_offsets();
  test_deterministic_repeat();
  test_empty_duplicate_missing();
  test_overflow_and_inconsistent();
  test_failure_cleanup();
  test_writer_interleaving_owns_publication();
  test_chunked_stream();
  if (g_failures != 0) {
    std::cerr << g_failures << " format writer checks failed\n";
    return 1;
  }
  std::cout << "format writer unit tests ok\n";
  return 0;
}
