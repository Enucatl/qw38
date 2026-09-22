#include "format/format.hpp"

#include <algorithm>
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

using qw38::format::ArithmeticDtype;
using qw38::format::ArtifactSchema;
using qw38::format::ArtifactWriter;
using qw38::format::decode_header;
using qw38::format::decode_schema;
using qw38::format::error_message;
using qw38::format::expected_payload_bytes;
using qw38::format::expected_scale_bytes;
using qw38::format::GraphBinding;
using qw38::format::Hash256;
using qw38::format::IntegrityKind;
using qw38::format::kNoLayerIndex;
using qw38::format::kSpanAlignment;
using qw38::format::LogicalPhysicalMapping;
using qw38::format::LogicalQuantizerId;
using qw38::format::MappingKind;
using qw38::format::PhysicalLayoutId;
using qw38::format::SemanticNodeKind;
using qw38::format::SemanticScope;
using qw38::format::sha256;
using qw38::format::SharedBinding;
using qw38::format::SharedBindingRole;
using qw38::format::SpanKind;
using qw38::format::StateAllocation;
using qw38::format::StateKind;
using qw38::format::StorageClass;
using qw38::format::TensorRecord;
using qw38::format::TensorRole;
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

TensorShape rank2(std::uint64_t n, std::uint64_t k) {
  TensorShape s{};
  s.rank = 2;
  s.logical[0] = n;
  s.logical[1] = k;
  s.padded[0] = n;
  s.padded[1] = k;
  return s;
}

TensorShape rank3(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
  TensorShape s{};
  s.rank = 3;
  s.logical[0] = a;
  s.logical[1] = b;
  s.logical[2] = c;
  s.padded[0] = a;
  s.padded[1] = b;
  s.padded[2] = c;
  return s;
}

std::vector<std::byte> pattern(std::uint64_t n, std::uint8_t seed) {
  std::vector<std::byte> out(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::byte>(static_cast<std::uint8_t>(seed + i));
  }
  return out;
}

std::vector<std::byte> read_all(std::filesystem::path const& path) {
  std::ifstream in(path, std::ios::binary);
  in.seekg(0, std::ios::end);
  auto const n = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<std::byte> bytes(n);
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(n));
  return bytes;
}

}  // namespace

int main() {
  auto tmpl_path =
      std::filesystem::temp_directory_path() / "qw38-writer-int-XXXXXX";
  std::string tmpl = tmpl_path.string();
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  if (::mkdtemp(buf.data()) == nullptr) {
    fail("mkdtemp");
    return 1;
  }
  std::filesystem::path dir{buf.data()};
  auto dest = dir / "fixture.qw38";

  ArtifactSchema schema{};
  schema.compiler = {
      .ident = "qw38-compiler", .major = 0, .minor = 1, .patch = 0};
  schema.source_hash.bytes[0] = 0x11;
  schema.config_hash.bytes[0] = 0x22;
  schema.tokenizer_hash.bytes[0] = 0x33;
  schema.precision = v0_precision_policy();
  schema.scope = SemanticScope::LanguagePlusMtpDescriptors;

  TensorRecord embed{};
  embed.tensor_id = 1;
  embed.logical_name = "model.embed_tokens.weight";
  embed.shape = rank2(8, 8);
  embed.storage = StorageClass::Bf16;
  embed.quantizer = LogicalQuantizerId::None;
  embed.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  embed.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};

  TensorRecord q4{};
  q4.tensor_id = 2;
  q4.logical_name = "model.layers.0.mlp.down_proj.weight";
  q4.shape = rank2(8, 256);
  q4.storage = StorageClass::Int4Grouped;
  q4.quantizer = LogicalQuantizerId::Q4G64V0;
  q4.layout = PhysicalLayoutId::CudaQ4G64V0;
  q4.mapping = LogicalPhysicalMapping{
      .kind = MappingKind::DenseTileNK,
      .tile_rows = 8,
      .tile_k = 256,
      .group_size = 64,
      .packed_bytes_per_tile_row = 128,
  };

  TensorRecord q8{};
  q8.tensor_id = 3;
  q8.logical_name = "lm_head.weight";
  q8.shape = rank2(8, 256);
  q8.storage = StorageClass::Int8Grouped;
  q8.quantizer = LogicalQuantizerId::Q8G32V0;
  q8.layout = PhysicalLayoutId::CudaQ8G32V0;
  q8.mapping = LogicalPhysicalMapping{
      .kind = MappingKind::DenseTileNK,
      .tile_rows = 8,
      .tile_k = 256,
      .group_size = 32,
      .packed_bytes_per_tile_row = 256,
  };

  TensorRecord mtp_embed = embed;
  mtp_embed.tensor_id = 4;
  mtp_embed.logical_name = "mtp.embed_tokens.weight";

  schema.tensors = {embed, q4, q8, mtp_embed};
  schema.graph_bindings = {
      GraphBinding{.instance_id = 1,
                   .kind = SemanticNodeKind::Embed,
                   .role = TensorRole::EmbeddingTable,
                   .layer_index = kNoLayerIndex,
                   .tensor_id = 1},
      GraphBinding{.instance_id = 2,
                   .kind = SemanticNodeKind::Mlp,
                   .role = TensorRole::DenseWeight,
                   .layer_index = 0,
                   .tensor_id = 2},
      GraphBinding{.instance_id = 3,
                   .kind = SemanticNodeKind::LmHead,
                   .role = TensorRole::LmHeadWeight,
                   .layer_index = kNoLayerIndex,
                   .tensor_id = 3},
  };
  schema.shared_bindings = {
      SharedBinding{.owner_tensor_id = 1,
                    .alias_tensor_id = 4,
                    .role = SharedBindingRole::MtpEmbeddingAlias},
  };

  StateAllocation gdn{};
  gdn.kind = StateKind::GdnS;
  gdn.dtype = ArithmeticDtype::Fp32;
  gdn.layout = PhysicalLayoutId::CudaFp32GdnSHvKV0;
  gdn.shape_per_layer = rank3(48, 128, 128);
  gdn.layer_count = 48;
  gdn.component_count = 1;
  gdn.bytes_per_layer = 3145728;
  gdn.total_bytes = 150994944;

  StateAllocation conv{};
  conv.kind = StateKind::ConvolutionHistory;
  conv.dtype = ArithmeticDtype::Bf16;
  conv.layout = PhysicalLayoutId::CudaBf16ConvHistoryV0;
  conv.shape_per_layer = rank2(3, 10240);
  conv.layer_count = 48;
  conv.component_count = 1;
  conv.bytes_per_layer = 61440;
  conv.total_bytes = 2949120;

  StateAllocation kv{};
  kv.kind = StateKind::KvCache;
  kv.dtype = ArithmeticDtype::Bf16;
  kv.layout = PhysicalLayoutId::CudaBf16KvCacheV0;
  kv.shape_per_layer = rank2(4, 256);
  kv.layer_count = 16;
  kv.component_count = 2;
  kv.declared_capacity = 4096;
  kv.bytes_per_token = 65536;
  kv.bytes_per_layer = 16777216;
  kv.total_bytes = 268435456;
  kv.populated_length_distinct_from_capacity = true;
  schema.state = {gdn, conv, kv};
  auto const scratch = qw38::format::v0_language_scratch_schema();
  schema.scratch.assign(scratch.begin(), scratch.end());

  auto embed_n = expected_payload_bytes(embed);
  auto q4_n = expected_payload_bytes(q4);
  auto q4_s = expected_scale_bytes(q4);
  auto q8_n = expected_payload_bytes(q8);
  auto q8_s = expected_scale_bytes(q8);
  if (!embed_n || !q4_n || !q4_s || !q8_n || !q8_s) {
    fail("expected span sizes");
    std::filesystem::remove_all(dir);
    return 1;
  }

  auto embed_bytes = pattern(*embed_n, 0x10);
  auto q4_payload = pattern(*q4_n, 0x40);
  auto q4_scales = pattern(*q4_s, 0x50);
  auto q8_payload = pattern(*q8_n, 0x80);
  auto q8_scales = pattern(*q8_s, 0x90);

  auto writer = ArtifactWriter::create(dest, schema);
  if (!writer) {
    fail(std::string("create: ") + error_message(writer.error()));
    std::filesystem::remove_all(dir);
    return 1;
  }
  auto write = [&](std::string_view name, SpanKind kind,
                   std::vector<std::byte> const& bytes) {
    auto st = writer->write_span(name, kind, bytes);
    if (!st) {
      fail(std::string("write ") + std::string(name) + ": " +
           error_message(st.error()));
    }
  };
  write("lm_head.weight", SpanKind::Payload, q8_payload);
  write("model.embed_tokens.weight", SpanKind::Payload, embed_bytes);
  write("model.layers.0.mlp.down_proj.weight", SpanKind::Scales, q4_scales);
  write("model.layers.0.mlp.down_proj.weight", SpanKind::Payload, q4_payload);
  write("lm_head.weight", SpanKind::Scales, q8_scales);

  auto id = writer->finalize();
  if (!id) {
    fail(std::string("finalize: ") + error_message(id.error()));
    std::filesystem::remove_all(dir);
    return 1;
  }

  auto file = read_all(dest);
  auto header = decode_header(std::span<std::byte const>{file.data(), 64});
  if (!header) {
    fail(std::string("header: ") + error_message(header.error()));
    std::filesystem::remove_all(dir);
    return 1;
  }
  auto decoded = decode_schema(std::span<std::byte const>{
      file.data() + header->manifest_offset,
      static_cast<std::size_t>(header->manifest_length)});
  if (!decoded) {
    fail(std::string("schema: ") + error_message(decoded.error()));
    std::filesystem::remove_all(dir);
    return 1;
  }

  expect(decoded->tensors.size() == 4, "four tensor records");
  expect(decoded->shared_bindings.size() == 1, "one shared binding");
  expect(decoded->shared_bindings[0].role ==
             SharedBindingRole::MtpEmbeddingAlias,
         "MTP embedding alias");
  expect(decoded->tensors[0].payload == decoded->tensors[3].payload,
         "alias reuses embed payload span");
  expect(decoded->state.size() == 3, "GDN/conv/KV state schema");
  expect(decoded->state[0].kind == StateKind::GdnS &&
             decoded->state[0].total_bytes == 150994944,
         "GDN S schema");
  expect(!decoded->state[0].live_payload_present, "no live state");

  expect(decoded->tensors[0].layout == PhysicalLayoutId::CudaBf16RowMajorV0,
         "BF16 embed layout");
  expect(decoded->tensors[1].quantizer == LogicalQuantizerId::Q4G64V0 &&
             !decoded->tensors[1].scales.empty(),
         "Q4 code+scale");
  expect(decoded->tensors[2].quantizer == LogicalQuantizerId::Q8G32V0 &&
             !decoded->tensors[2].scales.empty(),
         "Q8 code+scale");

  for (auto const& t : decoded->tensors) {
    expect(t.payload.offset % kSpanAlignment == 0, "payload aligned");
    if (!t.scales.empty()) {
      expect(t.scales.offset % kSpanAlignment == 0, "scale aligned");
      expect(t.payload.offset != t.scales.offset,
             "payload and scale bases are separate");
    }
  }

  bool saw_manifest = false;
  int payload_hashes = 0;
  int scale_hashes = 0;
  for (auto const& rec : decoded->integrity) {
    Hash256 independent{};
    if (rec.kind == IntegrityKind::Sha256Manifest) {
      std::vector<std::byte> canonical(
          file.begin() + static_cast<std::ptrdiff_t>(rec.region.offset),
          file.begin() + static_cast<std::ptrdiff_t>(rec.region.offset +
                                                     rec.region.length));
      std::fill(canonical.end() - qw38::format::kHashBytes, canonical.end(),
                std::byte{});
      independent = sha256(canonical);
    } else {
      auto region = std::span<std::byte const>{
          file.data() + rec.region.offset,
          static_cast<std::size_t>(rec.region.length)};
      independent = sha256(region);
    }
    expect(independent == rec.digest,
           "integrity digest matches independent SHA-256 of declared region");
    if (rec.kind == IntegrityKind::Sha256Manifest) {
      saw_manifest = true;
      expect(rec.region.offset == header->manifest_offset, "manifest region");
    } else if (rec.kind == IntegrityKind::Sha256PayloadSpan) {
      ++payload_hashes;
    } else if (rec.kind == IntegrityKind::Sha256ScaleSpan) {
      ++scale_hashes;
    }
  }
  expect(saw_manifest, "manifest integrity record present");
  expect(payload_hashes == 4, "payload hashes for embed, q4, q8, and alias");
  expect(scale_hashes == 2, "scale hashes for q4 and q8");

  auto const& embed_span = decoded->tensors[0].payload;
  expect(std::equal(embed_bytes.begin(), embed_bytes.end(),
                    file.begin() + static_cast<std::ptrdiff_t>(embed_span.offset)),
         "embed payload bytes");
  auto const& q4_span = decoded->tensors[1].payload;
  expect(std::equal(q4_payload.begin(), q4_payload.end(),
                    file.begin() + static_cast<std::ptrdiff_t>(q4_span.offset)),
         "q4 payload bytes");
  auto const& q4_scale_span = decoded->tensors[1].scales;
  expect(std::equal(q4_scales.begin(), q4_scales.end(),
                    file.begin() +
                        static_cast<std::ptrdiff_t>(q4_scale_span.offset)),
         "q4 scale bytes");

  std::cout << "format writer integration ok: dest=" << dest
            << " bytes=" << file.size()
            << " manifest_off=" << header->manifest_offset
            << " tensors=" << decoded->tensors.size() << '\n';
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  if (g_failures != 0) {
    std::cerr << g_failures << " writer integration checks failed\n";
    return 1;
  }
  return 0;
}
