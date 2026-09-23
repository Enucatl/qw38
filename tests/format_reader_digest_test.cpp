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
  bool saw_payload_digest = false;
  bool saw_manifest = false;
  for (auto const& rec : art->schema().integrity) {
    if (rec.kind == IntegrityKind::Sha256PayloadSpan) {
      saw_payload_digest = true;
      fail("new artifacts must not carry payload digests");
    } else if (rec.kind == IntegrityKind::Sha256ScaleSpan) {
      fail("new artifacts must not carry scale digests");
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
    }
  }
  expect(!saw_payload_digest && saw_manifest,
         "manifest-only digest is present");

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
    expect(false, "new artifacts must not carry payload or scale digests");
  }
  auto embed = multi->payload("model.embed_tokens.weight");
  auto alias = multi->payload("mtp.embed_tokens.weight");
  if (!embed || !alias) {
    fail("embed/alias payload lookup");
    return 1;
  }
  expect(std::equal(embed->begin(), embed->end(), alias->begin()),
         "alias payload equals owner payload");
  expect(std::equal(embed->begin(), embed->end(), fx.embed.begin()),
         "embed payload equals source bytes");
  auto q4p = multi->payload("model.layers.0.mlp.down_proj.weight");
  auto q4s = multi->scales("model.layers.0.mlp.down_proj.weight");
  if (!q4p || !q4s) {
    fail("q4 span lookup");
    return 1;
  }
  expect(std::equal(q4p->begin(), q4p->end(), fx.q4_payload.begin()),
         "q4 payload equals source bytes");
  expect(std::equal(q4s->begin(), q4s->end(), fx.q4_scales.begin()),
         "q4 scales equal source bytes");

  if (g_failures != 0) {
    std::cerr << g_failures << " reader digest checks failed\n";
    return 1;
  }
  std::cout << "format reader digest ok\n";
  return 0;
}
