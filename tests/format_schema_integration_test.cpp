#include "format/format.hpp"

#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using qw38::format::align_up;
using qw38::format::ArithmeticDtype;
using qw38::format::ArtifactSchema;
using qw38::format::ByteSpan;
using qw38::format::decode_schema;
using qw38::format::encode;
using qw38::format::encoded_size;
using qw38::format::error_message;
using qw38::format::expected_payload_bytes;
using qw38::format::expected_scale_bytes;
using qw38::format::GraphBinding;
using qw38::format::Hash256;
using qw38::format::IntegrityKind;
using qw38::format::IntegrityRecord;
using qw38::format::kNoLayerIndex;
using qw38::format::kSpanAlignment;
using qw38::format::LogicalPhysicalMapping;
using qw38::format::LogicalQuantizerId;
using qw38::format::MappingKind;
using qw38::format::PhysicalLayoutId;
using qw38::format::SemanticNodeKind;
using qw38::format::SemanticScope;
using qw38::format::StateAllocation;
using qw38::format::StateKind;
using qw38::format::StorageClass;
using qw38::format::TensorRecord;
using qw38::format::TensorRole;
using qw38::format::TensorShape;
using qw38::format::v0_precision_policy;
using qw38::format::validate_schema;

namespace {

Hash256 hash_with(std::uint8_t seed) {
  Hash256 h{};
  for (std::size_t i = 0; i < h.bytes.size(); ++i) {
    h.bytes[i] = static_cast<std::uint8_t>(seed + static_cast<std::uint8_t>(i));
  }
  return h;
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

bool fail(std::string_view what) {
  std::cerr << "FAIL: " << what << '\n';
  return false;
}

std::uint64_t place_span(std::uint64_t& cursor, std::uint64_t length) {
  auto aligned = align_up(cursor, kSpanAlignment, 0, "cursor");
  if (!aligned) {
    fail("span cursor overflow");
    return 0;
  }
  cursor = *aligned;
  auto end = cursor;
  auto added = qw38::format::checked_add(cursor, length, 0, "span");
  if (!added) {
    fail("span end overflow");
    return 0;
  }
  end = cursor;
  cursor = *added;
  return end;
}

}  // namespace

int main() {
  constexpr std::uint64_t kVocab = 248320;
  constexpr std::uint64_t kHidden = 5120;
  constexpr std::uint64_t kFfn = 17408;
  constexpr std::uint64_t kKvCapacity = 4096;

  ArtifactSchema schema{};
  schema.compiler = {
      .ident = "qw38-compiler", .major = 0, .minor = 1, .patch = 0};
  schema.source_hash = hash_with(0x10);
  schema.config_hash = hash_with(0x20);
  schema.tokenizer_hash = hash_with(0x30);
  schema.precision = v0_precision_policy();
  schema.scope = SemanticScope::PrimaryLanguage;

  TensorRecord embed{};
  embed.tensor_id = 1;
  embed.logical_name = "model.embed_tokens.weight";
  embed.shape = rank2(kVocab, kHidden);
  embed.storage = StorageClass::Bf16;
  embed.quantizer = LogicalQuantizerId::None;
  embed.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  embed.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};

  TensorRecord q4{};
  q4.tensor_id = 2;
  q4.logical_name = "model.layers.0.mlp.down_proj.weight";
  q4.shape = rank2(kHidden, kFfn);
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
  q8.shape = rank2(kVocab, kHidden);
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

  auto embed_n = expected_payload_bytes(embed);
  auto q4_n = expected_payload_bytes(q4);
  auto q4_s = expected_scale_bytes(q4);
  auto q8_n = expected_payload_bytes(q8);
  auto q8_s = expected_scale_bytes(q8);
  if (!embed_n || !q4_n || !q4_s || !q8_n || !q8_s) {
    return fail("expected span sizes") ? 1 : 1;
  }
  if (*embed_n != 2542796800ull || *q4_n != 44564480ull ||
      *q8_n != 1271398400ull) {
    return fail("representative payload sizes") ? 1 : 1;
  }

  std::uint64_t cursor = 4096;
  embed.payload = ByteSpan{.offset = place_span(cursor, *embed_n),
                           .length = *embed_n};
  q4.payload = ByteSpan{.offset = place_span(cursor, *q4_n), .length = *q4_n};
  q4.scales = ByteSpan{.offset = place_span(cursor, *q4_s), .length = *q4_s};
  q8.payload = ByteSpan{.offset = place_span(cursor, *q8_n), .length = *q8_n};
  q8.scales = ByteSpan{.offset = place_span(cursor, *q8_s), .length = *q8_s};
  schema.tensors = {embed, q4, q8};

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
  kv.declared_capacity = kKvCapacity;
  kv.bytes_per_token = 65536;
  kv.bytes_per_layer = 16777216;
  kv.total_bytes = 268435456;
  kv.populated_length_distinct_from_capacity = true;
  schema.state = {gdn, conv, kv};

  auto const scratch = qw38::format::v0_language_scratch_schema();
  schema.scratch.assign(scratch.begin(), scratch.end());

  schema.integrity = {
      IntegrityRecord{.kind = IntegrityKind::Sha256Manifest,
                      .region = ByteSpan{.offset = 64, .length = 128},
                      .digest = hash_with(0x40)},
      IntegrityRecord{.kind = IntegrityKind::Sha256PayloadSpan,
                      .tensor_id = 1,
                      .region = embed.payload,
                      .digest = hash_with(0x41)},
      IntegrityRecord{.kind = IntegrityKind::Sha256PayloadSpan,
                      .tensor_id = 2,
                      .region = q4.payload,
                      .digest = hash_with(0x42)},
      IntegrityRecord{.kind = IntegrityKind::Sha256ScaleSpan,
                      .tensor_id = 2,
                      .region = q4.scales,
                      .digest = hash_with(0x43)},
      IntegrityRecord{.kind = IntegrityKind::Sha256PayloadSpan,
                      .tensor_id = 3,
                      .region = q8.payload,
                      .digest = hash_with(0x44)},
      IntegrityRecord{.kind = IntegrityKind::Sha256ScaleSpan,
                      .tensor_id = 3,
                      .region = q8.scales,
                      .digest = hash_with(0x45)},
  };

  auto valid = validate_schema(schema);
  if (!valid) {
    std::cerr << error_message(valid.error()) << '\n';
    return fail("representative schema validation");
  }

  auto n = encoded_size(schema);
  if (!n) {
    std::cerr << error_message(n.error()) << '\n';
    return fail("encoded_size");
  }
  std::vector<std::byte> bytes(*n);
  auto enc = encode(schema, std::span<std::byte>{bytes});
  if (!enc) {
    std::cerr << error_message(enc.error()) << '\n';
    return fail("encode representative schema");
  }
  auto decoded = decode_schema(bytes);
  if (!decoded) {
    std::cerr << error_message(decoded.error()) << '\n';
    return fail("decode representative schema");
  }
  if (*decoded != schema) {
    return fail("roundtrip equality");
  }

  std::cout << "format schema integration ok: embed=" << *embed_n
            << " q4=" << *q4_n << " q4_scales=" << *q4_s << " q8=" << *q8_n
            << " q8_scales=" << *q8_s << " gdn_s=" << gdn.total_bytes
            << " conv=" << conv.total_bytes << " kv=" << kv.total_bytes
            << " wire_bytes=" << bytes.size() << '\n';
  return 0;
}
