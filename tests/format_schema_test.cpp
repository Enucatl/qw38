#include "format/format.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using qw38::format::ArtifactSchema;
using qw38::format::ByteSpan;
using qw38::format::ByteWriter;
using qw38::format::checked_add;
using qw38::format::checked_mul;
using qw38::format::align_up;
using qw38::format::ContainerHeader;
using qw38::format::decode_header;
using qw38::format::decode_physical_layout;
using qw38::format::decode_schema;
using qw38::format::encode;
using qw38::format::encoded_size;
using qw38::format::error_message;
using qw38::format::expected_payload_bytes;
using qw38::format::expected_scale_bytes;
using qw38::format::FormatErrorCode;
using qw38::format::GraphBinding;
using qw38::format::Hash256;
using qw38::format::IntegrityKind;
using qw38::format::IntegrityRecord;
using qw38::format::is_aligned;
using qw38::format::kHeaderSizeV0;
using qw38::format::kNoLayerIndex;
using qw38::format::kSpanAlignment;
using qw38::format::LogicalPhysicalMapping;
using qw38::format::LogicalQuantizerId;
using qw38::format::MappingKind;
using qw38::format::PhysicalLayoutId;
using qw38::format::SemanticNodeKind;
using qw38::format::SemanticScope;
using qw38::format::SharedBinding;
using qw38::format::SharedBindingRole;
using qw38::format::StorageClass;
using qw38::format::TensorRecord;
using qw38::format::TensorRole;
using qw38::format::TensorShape;
using qw38::format::v0_precision_policy;
using qw38::format::validate_schema;

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

Hash256 hash_with(std::uint8_t seed) {
  Hash256 h{};
  for (std::size_t i = 0; i < h.bytes.size(); ++i) {
    h.bytes[i] = static_cast<std::uint8_t>(seed + i);
  }
  return h;
}

TensorShape rank2(std::uint64_t n, std::uint64_t k, std::uint64_t pn,
                  std::uint64_t pk) {
  TensorShape s{};
  s.rank = 2;
  s.logical[0] = n;
  s.logical[1] = k;
  s.padded[0] = pn;
  s.padded[1] = pk;
  return s;
}

TensorShape rank1(std::uint64_t n) {
  TensorShape s{};
  s.rank = 1;
  s.logical[0] = n;
  s.padded[0] = n;
  return s;
}

LogicalPhysicalMapping identity_mapping() {
  return LogicalPhysicalMapping{.kind = MappingKind::Identity};
}

LogicalPhysicalMapping q4_mapping() {
  return LogicalPhysicalMapping{
      .kind = MappingKind::DenseTileNK,
      .tile_rows = 8,
      .tile_k = 256,
      .group_size = 64,
      .packed_bytes_per_tile_row = 128,
  };
}

LogicalPhysicalMapping q8_mapping() {
  return LogicalPhysicalMapping{
      .kind = MappingKind::DenseTileNK,
      .tile_rows = 8,
      .tile_k = 256,
      .group_size = 32,
      .packed_bytes_per_tile_row = 256,
  };
}

ArtifactSchema base_schema() {
  ArtifactSchema schema{};
  schema.compiler = {.ident = "qw38", .major = 0, .minor = 1, .patch = 0};
  schema.source_hash = hash_with(1);
  schema.config_hash = hash_with(2);
  schema.tokenizer_hash = hash_with(3);
  schema.precision = v0_precision_policy();
  schema.scope = SemanticScope::PrimaryLanguage;
  auto const state = qw38::format::v0_language_state_schema();
  schema.state.assign(state.begin(), state.end());
  auto const scratch = qw38::format::v0_language_scratch_schema();
  schema.scratch.assign(scratch.begin(), scratch.end());
  return schema;
}

TensorRecord bf16_vector(std::uint32_t id, std::string name, std::uint64_t n,
                         std::uint64_t offset) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = rank1(n);
  t.storage = StorageClass::Bf16;
  t.quantizer = LogicalQuantizerId::None;
  t.layout = PhysicalLayoutId::CudaBf16VectorV0;
  t.mapping = identity_mapping();
  t.payload = ByteSpan{.offset = offset, .length = n * 2};
  return t;
}

TensorRecord bf16_rows(std::uint32_t id, std::string name, std::uint64_t rows,
                       std::uint64_t cols, std::uint64_t offset) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = rank2(rows, cols, rows, cols);
  t.storage = StorageClass::Bf16;
  t.quantizer = LogicalQuantizerId::None;
  t.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  t.mapping = identity_mapping();
  t.payload = ByteSpan{.offset = offset, .length = rows * cols * 2};
  return t;
}

TensorRecord q4_matrix(std::uint32_t id, std::string name, std::uint64_t n,
                       std::uint64_t k, std::uint64_t payload_off,
                       std::uint64_t scale_off) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = rank2(n, k, n, k);
  t.storage = StorageClass::Int4Grouped;
  t.quantizer = LogicalQuantizerId::Q4G64V0;
  t.layout = PhysicalLayoutId::CudaQ4G64V0;
  t.mapping = q4_mapping();
  t.payload = ByteSpan{.offset = payload_off, .length = n * k / 2};
  t.scales = ByteSpan{.offset = scale_off, .length = n * (k / 64) * 2};
  return t;
}

std::vector<std::byte> must_encode(ArtifactSchema const& schema,
                                   std::string_view what) {
  auto const n = encoded_size(schema);
  if (!n) {
    fail(std::string(what) + ": encoded_size " + error_message(n.error()));
    return {};
  }
  std::vector<std::byte> bytes(*n);
  auto const st = encode(schema, bytes);
  if (!st) {
    fail(std::string(what) + ": encode " + error_message(st.error()));
    return {};
  }
  return bytes;
}

void test_golden_header_and_integers() {
  std::array<std::byte, 8> intbuf{};
  ByteWriter w{intbuf};
  expect(static_cast<bool>(w.u16(0x0201, "u16")), "u16 write");
  expect(static_cast<bool>(w.u32(0xA1B2C3D4u, "u32")), "u32 write");
  expect(static_cast<bool>(w.u8(0xEE, "u8")), "u8 write");
  expect(intbuf[0] == std::byte{0x01} && intbuf[1] == std::byte{0x02},
         "u16 little-endian 0x0201 -> 01 02");
  expect(intbuf[2] == std::byte{0xD4} && intbuf[3] == std::byte{0xC3} &&
             intbuf[4] == std::byte{0xB2} && intbuf[5] == std::byte{0xA1},
         "u32 little-endian 0xA1B2C3D4 -> D4 C3 B2 A1");
  expect(intbuf[6] == std::byte{0xEE}, "u8 0xEE");

  std::array<std::byte, 8> u64buf{};
  ByteWriter w64{u64buf};
  expect(static_cast<bool>(w64.u64(0x0102030405060708ull, "u64")), "u64 write");
  std::array<std::uint8_t, 8> const want_u64{0x08, 0x07, 0x06, 0x05,
                                             0x04, 0x03, 0x02, 0x01};
  for (std::size_t i = 0; i < 8; ++i) {
    expect(static_cast<std::uint8_t>(u64buf[i]) == want_u64[i],
           "u64 little-endian byte");
  }

  ContainerHeader header{};
  header.manifest_offset = 256;
  header.manifest_length = 16;
  std::array<std::byte, kHeaderSizeV0> raw{};
  auto const enc = encode(header, raw);
  expect(static_cast<bool>(enc), "header encode");
  std::array<std::uint8_t, 8> const magic{'Q', 'W', '3', '8', 'F', 'M', 'T', 0};
  for (std::size_t i = 0; i < 8; ++i) {
    expect(static_cast<std::uint8_t>(raw[i]) == magic[i], "header magic golden");
  }
  expect(static_cast<std::uint8_t>(raw[8]) == 1 &&
             static_cast<std::uint8_t>(raw[9]) == 0,
         "container version LE 1");
  expect(static_cast<std::uint8_t>(raw[10]) == 1 &&
             static_cast<std::uint8_t>(raw[11]) == 0,
         "manifest version LE 1");
  expect(static_cast<std::uint8_t>(raw[12]) == 64 &&
             static_cast<std::uint8_t>(raw[13]) == 0,
         "header size LE 64");
  expect(static_cast<std::uint8_t>(raw[14]) == 0 &&
             static_cast<std::uint8_t>(raw[15]) == 0,
         "header reserved0 zero");
  expect(static_cast<std::uint8_t>(raw[16]) == 0x00 &&
             static_cast<std::uint8_t>(raw[17]) == 0x01,
         "manifest offset 256 LE");
  expect(static_cast<std::uint8_t>(raw[24]) == 0x10 &&
             static_cast<std::uint8_t>(raw[25]) == 0x00,
         "manifest length 16 LE");
  for (std::size_t i = 32; i < raw.size(); ++i) {
    expect(raw[i] == std::byte{0}, "header tail reserved zero");
  }
  auto decoded = decode_header(raw);
  expect(static_cast<bool>(decoded) && *decoded == header, "header roundtrip");
}

void test_endian_mismatch_is_rejected_as_version() {
  ContainerHeader header{};
  header.manifest_offset = 64;
  header.manifest_length = 8;
  std::array<std::byte, kHeaderSizeV0> raw{};
  expect(static_cast<bool>(encode(header, raw)), "encode header for endian case");
  // Swap the two container-version bytes as if they were stored big-endian.
  std::swap(raw[8], raw[9]);
  auto decoded = decode_header(raw);
  expect(!decoded &&
             decoded.error().code == FormatErrorCode::UnsupportedContainerVersion,
         "big-endian version field is rejected");
}

void test_enum_and_version_rejection() {
  expect(std::to_underlying(LogicalQuantizerId::Q4G64V0) !=
             std::to_underlying(PhysicalLayoutId::CudaQ4G64V0),
         "quantizer and layout IDs occupy disjoint wire ranges");
  auto as_layout = decode_physical_layout(
      std::to_underlying(LogicalQuantizerId::Q4G64V0), 0, "layout");
  expect(!as_layout && as_layout.error().code == FormatErrorCode::UnknownEnum,
         "logical quantizer wire value is not a physical layout");

  ContainerHeader header{};
  header.container_version = 99;
  header.manifest_offset = 64;
  std::array<std::byte, kHeaderSizeV0> raw{};
  expect(static_cast<bool>(encode(header, raw)), "encode unsupported version");
  auto decoded = decode_header(raw);
  expect(!decoded &&
             decoded.error().code == FormatErrorCode::UnsupportedContainerVersion,
         "unsupported container version rejected");

  header = {};
  header.magic[0] = 'X';
  header.manifest_offset = 64;
  expect(static_cast<bool>(encode(header, raw)), "encode bad magic");
  decoded = decode_header(raw);
  expect(!decoded && decoded.error().code == FormatErrorCode::BadMagic,
         "bad magic rejected");

  auto schema = base_schema();
  schema.tensors.push_back(bf16_vector(1, "norm", 4, 256));
  auto bytes = must_encode(schema, "enum schema");
  if (bytes.empty()) {
    return;
  }
  // Corrupt the storage-class u16 inside the first tensor by scanning for the
  // known Int4Grouped/Bf16 value after the name. Safer: flip a high byte of
  // the first tensor storage field via a known layout. Re-encode a Q4 tensor
  // and poke the quantizer field to a layout ID.
  schema.tensors.clear();
  schema.tensors.push_back(q4_matrix(1, "w", 8, 256, 256, 1280));
  bytes = must_encode(schema, "q4 schema");
  if (bytes.empty()) {
    return;
  }
  auto ok = decode_schema(bytes);
  if (!ok) {
    fail(std::string("valid q4 schema decodes: ") + error_message(ok.error()));
  }

  schema.tensors[0].quantizer = LogicalQuantizerId::Q4G64V0;
  schema.tensors[0].layout = PhysicalLayoutId::CudaQ8G32V0;
  schema.tensors[0].mapping = q8_mapping();
  auto st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidQuantizerLayoutPair,
         "Q4 quantizer cannot use cuda_q8g32_v0");

  schema.tensors[0].layout = PhysicalLayoutId::CudaQ4G64V0;
  schema.tensors[0].mapping = q4_mapping();
  schema.tensors[0].storage = StorageClass::Bf16;
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidStorageQuantizerPair,
         "Q4 quantizer cannot use BF16 storage");
}

void test_overflow_and_alignment() {
  auto mul = checked_mul(1ull << 32, 1ull << 32, 12, "dims");
  expect(!mul && mul.error().code == FormatErrorCode::Overflow &&
             mul.error().offset == 12 && mul.error().field == "dims",
         "32x32-bit multiply overflow is typed");
  auto add = checked_add(std::numeric_limits<std::uint64_t>::max(), 1, 4, "end");
  expect(!add && add.error().code == FormatErrorCode::Overflow, "add overflow");
  auto aligned = align_up(std::numeric_limits<std::uint64_t>::max(), kSpanAlignment, 0, "align");
  expect(!aligned && aligned.error().code == FormatErrorCode::Overflow,
         "align_up overflow near UINT64_MAX");
  auto ok_align = align_up(1, kSpanAlignment, 0, "align");
  expect(static_cast<bool>(ok_align) && *ok_align == 256, "align_up 1 -> 256");
  expect(is_aligned(256, kSpanAlignment) && !is_aligned(128, kSpanAlignment),
         "256-byte alignment predicate");

  auto schema = base_schema();
  auto t = bf16_vector(1, "v", 4, 128);
  schema.tensors.push_back(t);
  auto st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::Misaligned,
         "unaligned payload base rejected");

  schema.tensors[0].payload.offset = 256;
  schema.tensors[0].payload.length = 8;
  st = validate_schema(schema);
  expect(static_cast<bool>(st), "aligned payload accepted");

  schema.tensors[0].payload = ByteSpan{.offset = 32, .length = 0};
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidSpan,
         "empty span must use offset 0");
}

void test_invalid_shape_and_span() {
  auto schema = base_schema();
  auto t = q4_matrix(1, "w", 8, 256, 256, 1280);
  t.shape.padded[0] = 7;
  schema.tensors.push_back(t);
  auto st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidShape,
         "padded N not multiple of 8 rejected");

  schema.tensors[0] = q4_matrix(1, "w", 8, 256, 256, 1280);
  schema.tensors[0].shape.padded[1] = 255;
  schema.tensors[0].shape.logical[1] = 255;
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidShape,
         "padded K not multiple of 256 rejected");

  schema.tensors[0] = q4_matrix(1, "w", 8, 256, 256, 1280);
  schema.tensors[0].shape.padded[0] = 4;
  schema.tensors[0].shape.logical[0] = 8;
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidShape,
         "padded < logical rejected");

  schema.tensors[0] = q4_matrix(1, "w", 8, 256, 256, 1280);
  schema.tensors[0].payload.length = 100;
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidSpan,
         "wrong Q4 payload length rejected");

  schema.tensors[0] = q4_matrix(1, "w", 8, 256, 256, 1280);
  schema.tensors[0].scales.length = 0;
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidSpan,
         "missing Q4 scales rejected");

  schema.tensors[0] = bf16_rows(1, "E", 2, 4, 256);
  schema.tensors[0].scales = ByteSpan{.offset = 512, .length = 32};
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::InvalidSpan,
         "BF16 tensor cannot carry a scale span");
}

void test_shape_entry_points_reject_malformed_ranks_and_sizes() {
  TensorRecord tensor{};
  tensor.logical_name = "shape-test";
  tensor.storage = StorageClass::Bf16;
  tensor.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  tensor.mapping = identity_mapping();

  tensor.shape.rank = 8;
  tensor.shape.logical.fill(1);
  tensor.shape.padded.fill(1);
  auto dims = tensor.shape.logical_dims();
  expect(dims && dims->size() == 8, "rank-8 logical_dims is valid");
  auto bytes = expected_payload_bytes(tensor);
  expect(bytes && *bytes == 2, "rank-8 payload sizing is valid");
  auto scales = expected_scale_bytes(tensor);
  expect(scales && *scales == 0, "rank-8 scale sizing is valid");

  tensor.shape.rank = 9;
  dims = tensor.shape.logical_dims();
  expect(!dims && dims.error().code == FormatErrorCode::InvalidShape,
         "rank-9 logical_dims returns InvalidShape");
  bytes = expected_payload_bytes(tensor);
  expect(!bytes && bytes.error().code == FormatErrorCode::InvalidShape,
         "rank-9 payload sizing returns InvalidShape");
  scales = expected_scale_bytes(tensor);
  expect(!scales && scales.error().code == FormatErrorCode::InvalidShape,
         "rank-9 scale sizing returns InvalidShape");
  auto schema = base_schema();
  schema.tensors.push_back(tensor);
  auto size = encoded_size(schema);
  expect(!size && size.error().code == FormatErrorCode::InvalidShape,
         "encoded_size rejects rank 9 before array access");
  std::array<std::byte, 4096> output{};
  auto encoded = encode(schema, output);
  expect(!encoded && encoded.error().code == FormatErrorCode::InvalidShape,
         "encode rejects rank 9 before array access");

  tensor.shape = {};
  dims = tensor.shape.padded_dims();
  expect(!dims && dims.error().code == FormatErrorCode::InvalidShape,
         "rank-0 padded_dims returns InvalidShape");
  bytes = expected_payload_bytes(tensor);
  expect(!bytes && bytes.error().code == FormatErrorCode::InvalidShape,
         "rank-0 payload sizing returns InvalidShape");

  tensor.shape = rank1(0);
  bytes = expected_payload_bytes(tensor);
  expect(!bytes && bytes.error().code == FormatErrorCode::InvalidShape,
         "zero extent payload sizing returns InvalidShape");

  tensor.shape = rank2(std::numeric_limits<std::uint64_t>::max(), 2,
                       std::numeric_limits<std::uint64_t>::max(), 2);
  bytes = expected_payload_bytes(tensor);
  expect(!bytes && bytes.error().code == FormatErrorCode::Overflow,
         "dimension product overflow is typed");
  schema.tensors[0] = tensor;
  size = encoded_size(schema);
  expect(!size && size.error().code == FormatErrorCode::Overflow,
         "schema sizing rejects dimension product overflow");
  encoded = encode(schema, output);
  expect(!encoded && encoded.error().code == FormatErrorCode::Overflow,
         "encode rejects dimension product overflow");
}

void test_shared_bindings() {
  auto schema = base_schema();
  auto embed = bf16_rows(1, "embed_tokens", 8, 8, 256);
  auto tied = embed;
  tied.tensor_id = 2;
  tied.logical_name = "lm_head";
  schema.tensors.push_back(embed);
  schema.tensors.push_back(tied);
  schema.graph_bindings.push_back(GraphBinding{
      .instance_id = 1,
      .kind = SemanticNodeKind::Embed,
      .role = TensorRole::EmbeddingTable,
      .layer_index = kNoLayerIndex,
      .tensor_id = 1,
  });
  schema.graph_bindings.push_back(GraphBinding{
      .instance_id = 2,
      .kind = SemanticNodeKind::LmHead,
      .role = TensorRole::LmHeadWeight,
      .layer_index = kNoLayerIndex,
      .tensor_id = 2,
  });
  schema.shared_bindings.push_back(SharedBinding{
      .owner_tensor_id = 1,
      .alias_tensor_id = 2,
      .role = SharedBindingRole::GenericAlias,
  });
  auto st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::SharedBinding,
         "tying embed and lm_head is rejected");

  schema = base_schema();
  auto a = bf16_vector(1, "a", 4, 256);
  auto b = a;
  b.tensor_id = 2;
  b.logical_name = "b";
  schema.tensors.push_back(a);
  schema.tensors.push_back(b);
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::SharedBinding,
         "identical spans without a shared binding are rejected");

  schema.shared_bindings.push_back(SharedBinding{
      .owner_tensor_id = 1,
      .alias_tensor_id = 2,
      .role = SharedBindingRole::GenericAlias,
  });
  st = validate_schema(schema);
  expect(static_cast<bool>(st), "generic alias of the same payload is valid");

  schema.shared_bindings[0].owner_tensor_id = 1;
  schema.shared_bindings[0].alias_tensor_id = 1;
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::SharedBinding,
         "self-binding is rejected");

  schema.shared_bindings[0].alias_tensor_id = 2;
  schema.shared_bindings[0].role = SharedBindingRole::MtpEmbeddingAlias;
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::SharedBinding,
         "MTP embedding alias requires MTP descriptor scope");

  schema.scope = SemanticScope::LanguagePlusMtpDescriptors;
  st = validate_schema(schema);
  expect(!st && st.error().code == FormatErrorCode::SharedBinding,
         "MTP embedding alias must point at the embedding table");

  schema = base_schema();
  schema.scope = SemanticScope::LanguagePlusMtpDescriptors;
  auto e = bf16_rows(1, "embed_tokens", 8, 8, 256);
  auto e_next = e;
  e_next.tensor_id = 2;
  e_next.logical_name = "mtp.embed";
  schema.tensors.push_back(e);
  schema.tensors.push_back(e_next);
  schema.graph_bindings.push_back(GraphBinding{
      .instance_id = 1,
      .kind = SemanticNodeKind::Embed,
      .role = TensorRole::EmbeddingTable,
      .layer_index = kNoLayerIndex,
      .tensor_id = 1,
  });
  schema.shared_bindings.push_back(SharedBinding{
      .owner_tensor_id = 1,
      .alias_tensor_id = 2,
      .role = SharedBindingRole::MtpEmbeddingAlias,
  });
  st = validate_schema(schema);
  expect(static_cast<bool>(st), "MTP embedding alias of the untied table is valid");
}

void test_shared_binding_ownership_graph() {
  auto make_schema = [] {
    auto schema = base_schema();
    auto a = bf16_vector(1, "a", 4, 256);
    auto b = a;
    b.tensor_id = 2;
    b.logical_name = "b";
    auto c = a;
    c.tensor_id = 3;
    c.logical_name = "c";
    schema.tensors = {a, b, c};
    return schema;
  };
  auto rejects = [](ArtifactSchema const& schema, std::string_view what) {
    auto st = validate_schema(schema);
    expect(!st && st.error().code == FormatErrorCode::SharedBinding, what);
  };

  auto schema = make_schema();
  schema.shared_bindings = {
      SharedBinding{.owner_tensor_id = 1, .alias_tensor_id = 2},
      SharedBinding{.owner_tensor_id = 2, .alias_tensor_id = 1},
  };
  rejects(schema, "reverse shared-binding pairs are rejected");

  schema = make_schema();
  schema.shared_bindings = {
      SharedBinding{.owner_tensor_id = 1, .alias_tensor_id = 3},
      SharedBinding{.owner_tensor_id = 2, .alias_tensor_id = 3},
  };
  rejects(schema, "an alias cannot have multiple owners");

  schema = make_schema();
  schema.shared_bindings = {
      SharedBinding{.owner_tensor_id = 1, .alias_tensor_id = 2},
      SharedBinding{.owner_tensor_id = 2, .alias_tensor_id = 3},
  };
  rejects(schema, "alias-as-owner chains are rejected");

  auto const n = encoded_size(schema);
  expect(static_cast<bool>(n), "invalid graph still has a wire size");
  if (n) {
    std::vector<std::byte> bytes(*n);
    expect(static_cast<bool>(encode(schema, bytes)),
           "invalid graph can be encoded for decoder rejection testing");
    auto decoded = decode_schema(bytes);
    expect(!decoded && decoded.error().code == FormatErrorCode::SharedBinding,
           "decoder rejects an encoded ownership chain");
  }

  schema = make_schema();
  schema.shared_bindings = {
      SharedBinding{.owner_tensor_id = 1, .alias_tensor_id = 2},
      SharedBinding{.owner_tensor_id = 2, .alias_tensor_id = 3},
      SharedBinding{.owner_tensor_id = 3, .alias_tensor_id = 1},
  };
  rejects(schema, "long shared-binding cycles are rejected");
}

void test_schema_roundtrip_and_object_layout_independence() {
  auto schema = base_schema();
  schema.tensors.push_back(bf16_vector(7, "dt_bias", 48, 256));
  schema.integrity.push_back(IntegrityRecord{
      .kind = IntegrityKind::Sha256Manifest,
      .tensor_id = 0,
      .region = ByteSpan{.offset = 64, .length = 128},
      .digest = hash_with(9),
  });
  auto bytes = must_encode(schema, "roundtrip");
  if (bytes.empty()) {
    return;
  }
  auto decoded = decode_schema(bytes);
  if (!decoded) {
    fail(std::string("schema encode/decode roundtrip: ") +
         error_message(decoded.error()));
  } else {
    expect(*decoded == schema, "schema encode/decode roundtrip");
  }

  // ABI bytes must not be a native struct dump: the in-memory TensorRecord
  // contains a std::string and is larger/pointer-bearing, while the wire form
  // is a length-prefixed name plus explicit LE fields.
  expect(sizeof(TensorRecord) != 12, "in-memory tensor is not a packed ABI blob");
  expect(bytes.size() > 8 && bytes[0] == std::byte{1} && bytes[1] == std::byte{0},
         "schema starts with LE manifest version 1");
}

void test_truncated_input() {
  auto schema = base_schema();
  schema.tensors.push_back(bf16_vector(1, "v", 4, 256));
  auto bytes = must_encode(schema, "truncate");
  if (bytes.size() < 4) {
    return;
  }
  bytes.resize(bytes.size() - 3);
  auto decoded = decode_schema(bytes);
  if (!decoded) {
    expect(decoded.error().code == FormatErrorCode::Truncated,
           std::string("truncated schema is a typed error: ") +
               error_message(decoded.error()));
  } else {
    fail("truncated schema is a typed error");
  }
}

void test_hostile_record_counts_are_bounded() {
  auto const schema = base_schema();
  auto const bytes = must_encode(schema, "hostile counts");
  if (bytes.empty()) {
    return;
  }

  auto put_u16 = [](std::vector<std::byte>& out, std::size_t offset,
                    std::uint16_t value) {
    out[offset] = static_cast<std::byte>(value & 0xFFu);
    out[offset + 1] = static_cast<std::byte>((value >> 8) & 0xFFu);
  };
  auto put_u32 = [](std::vector<std::byte>& out, std::size_t offset,
                    std::uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i) {
      out[offset + i] =
          static_cast<std::byte>((value >> (8 * i)) & 0xFFu);
    }
  };

  std::size_t const precision_count_offset =
      2 + (2 + schema.compiler.ident.size() + 16) + 3 * 32 + 2;
  auto hostile = bytes;
  put_u16(hostile, precision_count_offset,
          std::numeric_limits<std::uint16_t>::max());
  auto decoded = decode_schema(hostile);
  expect(!decoded &&
             decoded.error().code == FormatErrorCode::ResourceLimitExceeded,
         "UINT16_MAX precision count is bounded before reserve");

  std::size_t const tensor_count_offset =
      precision_count_offset + 2 + 4 * schema.precision.bindings.size() + 2;
  std::size_t state_bytes = 0;
  for (auto const& state : schema.state) {
    state_bytes += 48 + 8 + 16 * state.shape_per_layer.rank;
  }
  std::size_t const scratch_count_offset =
      tensor_count_offset + 4 + 4 + 4 + 4 + state_bytes;
  std::size_t const integrity_count_offset =
      scratch_count_offset + 4 + 16 * schema.scratch.size();
  std::array<std::size_t, 6> const count_offsets{
      tensor_count_offset, tensor_count_offset + 4, tensor_count_offset + 8,
      tensor_count_offset + 12, scratch_count_offset,
      integrity_count_offset};
  for (auto const offset : count_offsets) {
    hostile = bytes;
    put_u32(hostile, offset, std::numeric_limits<std::uint32_t>::max());
    decoded = decode_schema(hostile);
    expect(!decoded &&
               decoded.error().code == FormatErrorCode::ResourceLimitExceeded,
           "UINT32_MAX record count is bounded before reserve");
  }

  hostile = bytes;
  put_u32(hostile, tensor_count_offset, 100);
  decoded = decode_schema(hostile);
  expect(!decoded && decoded.error().code == FormatErrorCode::Truncated,
         "plausible count is preflighted against remaining manifest bytes");
}

}  // namespace

int main() {
  test_golden_header_and_integers();
  test_endian_mismatch_is_rejected_as_version();
  test_enum_and_version_rejection();
  test_overflow_and_alignment();
  test_invalid_shape_and_span();
  test_shape_entry_points_reject_malformed_ranks_and_sizes();
  test_shared_bindings();
  test_shared_binding_ownership_graph();
  test_schema_roundtrip_and_object_layout_independence();
  test_truncated_input();
  test_hostile_record_counts_are_bounded();
  if (g_failures != 0) {
    std::cerr << g_failures << " format schema checks failed\n";
    return 1;
  }
  std::cout << "format schema unit tests ok\n";
  return 0;
}
