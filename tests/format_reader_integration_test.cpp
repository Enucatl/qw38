#include "format_reader_support.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <string_view>

using qw38::format::ArithmeticDtype;
using qw38::format::Artifact;
using qw38::format::error_message;
using qw38::format::FormatErrorCode;
using qw38::format::SemanticScope;
using qw38::format::StateKind;
using qw38::format::test::mutate_schema;
using qw38::format::test::read_all;
using qw38::format::test::ScratchDir;
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

void expect_code(std::expected<Artifact, qw38::format::FormatError> const& r,
                 FormatErrorCode code, std::string_view what) {
  if (r) {
    fail(std::string(what) + ": unexpectedly succeeded");
    return;
  }
  if (r.error().code != code) {
    fail(std::string(what) + ": got " + error_message(r.error()));
  }
}

void test_task003_roundtrip() {
  ScratchDir dir("qw38-reader-int");
  auto fx = write_task003(dir.file("fixture.qw38"));
  if (fx.path.empty()) {
    fail("write TASK-003 fixture");
    return;
  }
  auto art = Artifact::open(fx.path);
  if (!art) {
    fail(std::string("open: ") + error_message(art.error()));
    return;
  }

  expect(art->compiler().ident == "qw38-compiler", "compiler ident");
  expect(art->compiler().major == 0 && art->compiler().minor == 1 &&
             art->compiler().patch == 0,
         "compiler revision");
  expect(art->schema().source_hash.bytes[0] == 0x11, "source hash");
  expect(art->schema().config_hash.bytes[0] == 0x22, "config hash");
  expect(art->schema().tokenizer_hash.bytes[0] == 0x33, "tokenizer hash");
  expect(art->scope() == SemanticScope::LanguagePlusMtpDescriptors,
         "semantic scope");
  expect(art->precision().bindings.size() == 10, "V0 precision domains");
  expect(art->tensors().size() == 4, "four tensor records");
  expect(art->shared_bindings().size() == 1, "MTP shared binding");
  expect(art->graph_bindings().size() == 3, "graph bindings");
  expect(art->state().size() == 3, "GDN/conv/KV schema");
  expect(art->scratch().size() == 7, "complete V0 scratch schema");

  auto const* embed = art->find_tensor("model.embed_tokens.weight");
  auto const* q4 = art->find_tensor("model.layers.0.mlp.down_proj.weight");
  auto const* q8 = art->find_tensor("lm_head.weight");
  auto const* mtp = art->find_tensor("mtp.embed_tokens.weight");
  expect(embed && q4 && q8 && mtp, "logical identities resolve");
  expect(embed->layout == qw38::format::PhysicalLayoutId::CudaBf16RowMajorV0,
         "BF16 embed layout");
  expect(q4->quantizer == qw38::format::LogicalQuantizerId::Q4G64V0,
         "Q4 quantizer");
  expect(q8->quantizer == qw38::format::LogicalQuantizerId::Q8G32V0,
         "Q8 quantizer");
  expect(embed->payload == mtp->payload, "alias shares payload span");

  auto embed_bytes = art->payload("model.embed_tokens.weight");
  auto alias_bytes = art->payload("mtp.embed_tokens.weight");
  auto q4p = art->payload("model.layers.0.mlp.down_proj.weight");
  auto q4s = art->scales("model.layers.0.mlp.down_proj.weight");
  auto q8p = art->payload("lm_head.weight");
  auto q8s = art->scales("lm_head.weight");
  if (!embed_bytes || !alias_bytes || !q4p || !q4s || !q8p || !q8s) {
    fail("span lookup failed");
    return;
  }
  expect(std::equal(embed_bytes->begin(), embed_bytes->end(), fx.embed.begin()),
         "embed span bytes");
  expect(std::equal(alias_bytes->begin(), alias_bytes->end(), fx.embed.begin()),
         "alias span bytes");
  expect(std::equal(q4p->begin(), q4p->end(), fx.q4_payload.begin()),
         "q4 payload bytes");
  expect(std::equal(q4s->begin(), q4s->end(), fx.q4_scales.begin()),
         "q4 scale bytes");
  expect(std::equal(q8p->begin(), q8p->end(), fx.q8_payload.begin()),
         "q8 payload bytes");
  expect(std::equal(q8s->begin(), q8s->end(), fx.q8_scales.begin()),
         "q8 scale bytes");
  expect(embed_bytes->data() == alias_bytes->data(),
         "alias view aliases the same mapping");

  expect(art->state()[0].kind == StateKind::GdnS &&
             art->state()[0].total_bytes == 150994944 &&
             !art->state()[0].live_payload_present,
         "GDN S is language-only coefficients, not live state");
  expect(art->state()[1].kind == StateKind::ConvolutionHistory &&
             !art->state()[1].live_payload_present,
         "convolution history schema only");
  expect(art->state()[2].kind == StateKind::KvCache &&
             art->state()[2].populated_length_distinct_from_capacity &&
             !art->state()[2].live_payload_present,
         "KV capacity coefficients, not live cache");
}

void test_corruption_families() {
  ScratchDir dir("qw38-reader-corrupt");
  auto fx = write_task003(dir.file("c.qw38"));
  auto file = read_all(fx.path);

  auto precision = mutate_schema(file, [](auto& schema) {
    schema.precision.bindings[0].dtype = ArithmeticDtype::Bf16;
  });
  if (!precision) {
    fail(error_message(precision.error()));
    return;
  }
  expect_code(Artifact::parse(*precision),
              FormatErrorCode::InvalidPrecisionPolicy, "precision policy");

  auto graph = mutate_schema(file, [](auto& schema) {
    schema.graph_bindings[1].layer_index = qw38::format::kNoLayerIndex;
  });
  if (!graph) {
    fail(error_message(graph.error()));
    return;
  }
  expect_code(Artifact::parse(*graph), FormatErrorCode::InvalidGraphBinding,
              "graph binding");

  auto compiler = mutate_schema(file, [](auto& schema) {
    schema.compiler.ident.clear();
  });
  if (!compiler) {
    fail(error_message(compiler.error()));
    return;
  }
  expect_code(Artifact::parse(*compiler), FormatErrorCode::InvalidHeader,
              "empty compiler ident");

  auto scope = mutate_schema(file, [](auto& schema) {
    schema.scope = SemanticScope::PrimaryLanguage;
  });
  if (!scope) {
    fail(error_message(scope.error()));
    return;
  }
  expect_code(Artifact::parse(*scope), FormatErrorCode::SharedBinding,
              "MTP alias without MTP scope");

  auto missing = mutate_schema(file, [](auto& schema) {
    schema.integrity.clear();
  });
  if (!missing) {
    fail(error_message(missing.error()));
    return;
  }
  expect_code(Artifact::parse(*missing), FormatErrorCode::MissingIntegrity,
              "missing integrity records");

  auto manifest_region = mutate_schema(file, [](auto& schema) {
    for (auto& rec : schema.integrity) {
      if (rec.kind == qw38::format::IntegrityKind::Sha256Manifest) {
        rec.region.length = schema.integrity.empty() ? 0 : rec.region.length + 56;
      }
    }
  });
  if (!manifest_region) {
    fail(error_message(manifest_region.error()));
    return;
  }
  auto parsed_region = Artifact::parse(*manifest_region);
  if (parsed_region) {
    fail("hashing the full manifest including digest record was accepted");
  } else {
    expect(parsed_region.error().code == FormatErrorCode::InvalidSpan ||
               parsed_region.error().code ==
                   FormatErrorCode::IntegrityDigestMismatch,
           "manifest region must be prefix excluding the digest record");
  }

  auto kv = mutate_schema(file, [](auto& schema) {
    schema.state[2].bytes_per_token = 1;
  });
  if (!kv) {
    fail(error_message(kv.error()));
    return;
  }
  expect_code(Artifact::parse(*kv), FormatErrorCode::InvalidStateAllocation,
              "KV coefficient mismatch");

  auto domain = mutate_schema(file, [](auto& schema) {
    schema.precision.bindings.pop_back();
  });
  if (!domain) {
    fail(error_message(domain.error()));
    return;
  }
  expect_code(Artifact::parse(*domain), FormatErrorCode::InvalidPrecisionPolicy,
              "missing precision domain");
}

}  // namespace

int main() {
  test_task003_roundtrip();
  test_corruption_families();
  if (g_failures != 0) {
    std::cerr << g_failures << " reader integration checks failed\n";
    return 1;
  }
  std::cout << "format reader integration ok\n";
  return 0;
}
