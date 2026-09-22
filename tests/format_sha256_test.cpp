#include "format/format.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using qw38::format::ArtifactSchema;
using qw38::format::ArtifactWriter;
using qw38::format::decode_header;
using qw38::format::decode_schema;
using qw38::format::error_message;
using qw38::format::Hash256;
using qw38::format::IntegrityKind;
using qw38::format::LogicalPhysicalMapping;
using qw38::format::LogicalQuantizerId;
using qw38::format::MappingKind;
using qw38::format::PhysicalLayoutId;
using qw38::format::sha256;
using qw38::format::SpanKind;
using qw38::format::StorageClass;
using qw38::format::TensorRecord;
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

Hash256 parse_hex(std::string_view hex) {
  Hash256 h{};
  auto nibble = [](char c) -> std::uint8_t {
    if (c >= '0' && c <= '9') {
      return static_cast<std::uint8_t>(c - '0');
    }
    if (c >= 'a' && c <= 'f') {
      return static_cast<std::uint8_t>(c - 'a' + 10);
    }
    return 0;
  };
  for (std::size_t i = 0; i < 32; ++i) {
    h.bytes[i] = static_cast<std::uint8_t>((nibble(hex[i * 2]) << 4) |
                                          nibble(hex[i * 2 + 1]));
  }
  return h;
}

std::span<std::byte const> as_bytes(std::string_view s) {
  return std::span<std::byte const>{
      reinterpret_cast<std::byte const*>(s.data()), s.size()};
}

void test_known_vectors() {
  expect(sha256({}) ==
             parse_hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b"
                       "7852b855"),
         "SHA-256 of empty message");
  expect(sha256(as_bytes("abc")) ==
             parse_hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61"
                       "f20015ad"),
         "SHA-256 of abc");
  expect(sha256(as_bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
             parse_hex("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd4"
                       "19db06c1"),
         "SHA-256 of 56-byte NIST vector");
  std::vector<std::byte> million(1000000, std::byte{'a'});
  expect(sha256(million) ==
             parse_hex("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39cc"
                       "c7112cd0"),
         "SHA-256 of one million a characters");
}

void test_emitted_span_digest() {
  auto base = std::filesystem::temp_directory_path() / "qw38-sha256-XXXXXX";
  std::string tmpl = base.string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  if (::mkdtemp(buf.data()) == nullptr) {
    fail("mkdtemp");
    return;
  }
  std::filesystem::path dir{buf.data()};
  auto dest = dir / "span.qw38";

  ArtifactSchema schema{};
  schema.compiler = {.ident = "qw38", .major = 0, .minor = 1, .patch = 0};
  schema.precision = v0_precision_policy();
  auto const state = qw38::format::v0_language_state_schema();
  schema.state.assign(state.begin(), state.end());
  auto const scratch = qw38::format::v0_language_scratch_schema();
  schema.scratch.assign(scratch.begin(), scratch.end());
  TensorRecord t{};
  t.tensor_id = 1;
  t.logical_name = "v";
  t.shape.rank = 1;
  t.shape.logical[0] = 4;
  t.shape.padded[0] = 4;
  t.storage = StorageClass::Bf16;
  t.quantizer = LogicalQuantizerId::None;
  t.layout = PhysicalLayoutId::CudaBf16VectorV0;
  t.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};
  schema.tensors.push_back(t);

  auto writer = ArtifactWriter::create(dest, schema);
  if (!writer) {
    fail(std::string("create: ") + error_message(writer.error()));
    std::filesystem::remove_all(dir);
    return;
  }
  std::array<std::byte, 8> payload{
      std::byte{0x10}, std::byte{0x20}, std::byte{0x30}, std::byte{0x40},
      std::byte{0x50}, std::byte{0x60}, std::byte{0x70}, std::byte{0x80}};
  if (auto st = writer->write_span("v", SpanKind::Payload, payload); !st) {
    fail(std::string("write: ") + error_message(st.error()));
    std::filesystem::remove_all(dir);
    return;
  }
  auto id = writer->finalize();
  if (!id) {
    fail(std::string("finalize: ") + error_message(id.error()));
    std::filesystem::remove_all(dir);
    return;
  }

  std::ifstream in(dest, std::ios::binary);
  in.seekg(0, std::ios::end);
  auto const n = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<std::byte> bytes(n);
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(n));

  auto header = decode_header(std::span<std::byte const>{bytes.data(), 64});
  if (!header) {
    fail("decode header");
    std::filesystem::remove_all(dir);
    return;
  }
  auto manifest = decode_schema(std::span<std::byte const>{
      bytes.data() + header->manifest_offset,
      static_cast<std::size_t>(header->manifest_length)});
  if (!manifest) {
    fail(std::string("decode schema: ") + error_message(manifest.error()));
    std::filesystem::remove_all(dir);
    return;
  }
  Hash256 const independent = sha256(payload);
  bool found = false;
  for (auto const& rec : manifest->integrity) {
    if (rec.kind == IntegrityKind::Sha256PayloadSpan && rec.tensor_id == 1) {
      found = true;
      expect(rec.digest == independent,
             "emitted payload digest matches independent SHA-256");
      expect(rec.region.offset == 256 && rec.region.length == 8,
             "payload integrity region");
    }
  }
  expect(found, "payload integrity record present");
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

}  // namespace

int main() {
  test_known_vectors();
  test_emitted_span_digest();
  if (g_failures != 0) {
    std::cerr << g_failures << " SHA-256 checks failed\n";
    return 1;
  }
  std::cout << "format sha256 tests ok\n";
  return 0;
}
