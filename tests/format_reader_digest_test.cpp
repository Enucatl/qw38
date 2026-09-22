#include "format_reader_support.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using qw38::format::Artifact;
using qw38::format::error_message;
using qw38::format::Hash256;
using qw38::format::IntegrityKind;
using qw38::format::kIntegrityRecordBytes;
using qw38::format::sha256;
using qw38::format::test::read_all;
using qw38::format::test::ScratchDir;
using qw38::format::test::write_minimal;
using qw38::format::test::write_task003;

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

}  // namespace

int main() {
  ScratchDir dir("qw38-reader-digest");
  auto min = write_minimal(dir.file("min.qw38"));
  if (min.path.empty()) {
    fail("write minimal");
    return 1;
  }
  auto art = Artifact::open(min.path);
  if (!art) {
    fail(error_message(art.error()));
    return 1;
  }
  auto payload = art->payload("v");
  if (!payload) {
    fail(error_message(payload.error()));
    return 1;
  }
  Hash256 const independent_payload = sha256(*payload);
  Hash256 const from_source = sha256(min.payload);
  expect(independent_payload == from_source,
         "reader span hash matches source bytes");

  bool saw_payload = false;
  bool saw_manifest = false;
  for (auto const& rec : art->schema().integrity) {
    std::span<std::byte const> region;
    if (rec.kind == IntegrityKind::Sha256PayloadSpan) {
      region = *payload;
      saw_payload = true;
    } else if (rec.kind == IntegrityKind::Sha256Manifest) {
      auto const bytes = read_all(min.path);
      expect(rec.region.length ==
                 art->header().manifest_length,
             "manifest digest region is the complete manifest");
      std::vector<std::byte> canonical(
          bytes.begin() + static_cast<std::ptrdiff_t>(rec.region.offset),
          bytes.begin() + static_cast<std::ptrdiff_t>(rec.region.offset +
                                                      rec.region.length));
      std::fill(canonical.end() - qw38::format::kHashBytes, canonical.end(),
                std::byte{});
      expect(sha256(canonical) == rec.digest,
             "independent SHA-256 zeroes only self-digest bytes");
      saw_manifest = true;
      continue;
    } else {
      fail("unexpected integrity kind on minimal artifact");
      continue;
    }
    expect(sha256(region) == rec.digest,
           "independent SHA-256 matches stored digest");
  }
  expect(saw_payload && saw_manifest, "payload and manifest digests present");

  auto fx = write_task003(dir.file("multi.qw38"));
  auto multi = Artifact::open(fx.path);
  if (!multi) {
    fail(std::string("open multi: ") + error_message(multi.error()));
    return 1;
  }
  auto file = read_all(fx.path);
  for (auto const& rec : multi->schema().integrity) {
    if (rec.kind == IntegrityKind::Sha256Manifest) {
      std::vector<std::byte> canonical(
          file.begin() + static_cast<std::ptrdiff_t>(rec.region.offset),
          file.begin() + static_cast<std::ptrdiff_t>(rec.region.offset +
                                                     rec.region.length));
      std::fill(canonical.end() - qw38::format::kHashBytes, canonical.end(),
                std::byte{});
      expect(sha256(canonical) == rec.digest,
             "multi manifest digest zeroes only self-digest bytes");
      continue;
    }
    auto region = std::span<std::byte const>{
        file.data() + static_cast<std::size_t>(rec.region.offset),
        static_cast<std::size_t>(rec.region.length)};
    expect(sha256(region) == rec.digest,
           "independent digest of declared region");
  }
  auto embed = multi->payload("model.embed_tokens.weight");
  auto alias = multi->payload("mtp.embed_tokens.weight");
  if (!embed || !alias) {
    fail("embed/alias payload lookup");
    return 1;
  }
  expect(sha256(*embed) == sha256(*alias), "alias digest matches owner");
  expect(sha256(*embed) == sha256(fx.embed), "embed digest matches source");
  auto q4p = multi->payload("model.layers.0.mlp.down_proj.weight");
  auto q4s = multi->scales("model.layers.0.mlp.down_proj.weight");
  if (!q4p || !q4s) {
    fail("q4 span lookup");
    return 1;
  }
  expect(sha256(*q4p) == sha256(fx.q4_payload),
         "q4 payload digest matches source");
  expect(sha256(*q4s) == sha256(fx.q4_scales),
         "q4 scale digest matches source");

  if (g_failures != 0) {
    std::cerr << g_failures << " reader digest checks failed\n";
    return 1;
  }
  std::cout << "format reader digest ok\n";
  return 0;
}
