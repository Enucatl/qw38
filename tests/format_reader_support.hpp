#pragma once

#include "format/format.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace qw38::format::test {

inline TensorShape rank1(std::uint64_t n) {
  TensorShape s{};
  s.rank = 1;
  s.logical[0] = n;
  s.padded[0] = n;
  return s;
}

inline TensorShape rank2(std::uint64_t n, std::uint64_t k) {
  TensorShape s{};
  s.rank = 2;
  s.logical[0] = n;
  s.logical[1] = k;
  s.padded[0] = n;
  s.padded[1] = k;
  return s;
}

inline TensorShape rank3(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
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

inline std::vector<std::byte> pattern(std::uint64_t n, std::uint8_t seed) {
  std::vector<std::byte> out(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::byte>(static_cast<std::uint8_t>(seed + i));
  }
  return out;
}

inline std::vector<std::byte> read_all(std::filesystem::path const& path) {
  std::ifstream in(path, std::ios::binary);
  in.seekg(0, std::ios::end);
  auto const n = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<std::byte> bytes(n);
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(n));
  return bytes;
}

inline void write_all(std::filesystem::path const& path,
                      std::span<std::byte const> bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<char const*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

class ScratchDir {
 public:
  explicit ScratchDir(std::string_view prefix) {
    auto base = std::filesystem::temp_directory_path() /
                (std::string(prefix) + "-XXXXXX");
    std::string tmpl = base.string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (::mkdtemp(buf.data()) == nullptr) {
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

  [[nodiscard]] std::filesystem::path const& path() const { return path_; }
  [[nodiscard]] std::filesystem::path file(std::string_view name) const {
    return path_ / std::string(name);
  }

 private:
  std::filesystem::path path_;
};

inline ArtifactSchema base_schema() {
  ArtifactSchema schema{};
  schema.compiler = {.ident = "qw38", .major = 0, .minor = 1, .patch = 0};
  schema.precision = v0_precision_policy();
  auto const state = v0_language_state_schema();
  schema.state.assign(state.begin(), state.end());
  auto const scratch = v0_language_scratch_schema();
  schema.scratch.assign(scratch.begin(), scratch.end());
  return schema;
}

inline TensorRecord unplaced_bf16_vector(std::uint32_t id, std::string name,
                                         std::uint64_t n) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = rank1(n);
  t.storage = StorageClass::Bf16;
  t.quantizer = LogicalQuantizerId::None;
  t.layout = PhysicalLayoutId::CudaBf16VectorV0;
  t.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};
  return t;
}

struct MinimalFixture {
  std::filesystem::path path;
  std::array<std::byte, 8> payload{
      std::byte{0x00}, std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
      std::byte{0x44}, std::byte{0x55}, std::byte{0x66}, std::byte{0x77}};
};

inline MinimalFixture write_minimal(std::filesystem::path dest) {
  MinimalFixture fx;
  fx.path = std::move(dest);
  auto schema = base_schema();
  schema.tensors.push_back(unplaced_bf16_vector(1, "v", 4));
  auto writer = ArtifactWriter::create(fx.path, schema);
  if (!writer) {
    fx.path.clear();
    return fx;
  }
  auto wr = writer->write_span("v", SpanKind::Payload, fx.payload);
  if (!wr) {
    fx.path.clear();
    return fx;
  }
  auto id = writer->finalize();
  if (!id) {
    fx.path.clear();
    return fx;
  }
  return fx;
}

struct Task003Fixture {
  std::filesystem::path path;
  std::vector<std::byte> embed;
  std::vector<std::byte> q4_payload;
  std::vector<std::byte> q4_scales;
  std::vector<std::byte> q8_payload;
  std::vector<std::byte> q8_scales;
};

inline ArtifactSchema task003_schema() {
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
  auto const scratch = v0_language_scratch_schema();
  schema.scratch.assign(scratch.begin(), scratch.end());
  return schema;
}

template <typename Mutator>
inline Task003Fixture write_task003_mutated(std::filesystem::path dest,
                                            Mutator&& mutate) {
  Task003Fixture fx;
  fx.path = std::move(dest);
  auto schema = task003_schema();
  auto embed_n = expected_payload_bytes(schema.tensors[0]);
  auto q4_n = expected_payload_bytes(schema.tensors[1]);
  auto q4_s = expected_scale_bytes(schema.tensors[1]);
  auto q8_n = expected_payload_bytes(schema.tensors[2]);
  auto q8_s = expected_scale_bytes(schema.tensors[2]);
  if (!embed_n || !q4_n || !q4_s || !q8_n || !q8_s) {
    fx.path.clear();
    return fx;
  }
  fx.embed = pattern(*embed_n, 0x10);
  fx.q4_payload.assign(static_cast<std::size_t>(*q4_n), std::byte{0x11});
  fx.q4_scales.assign(static_cast<std::size_t>(*q4_s), std::byte{0});
  fx.q8_payload.assign(static_cast<std::size_t>(*q8_n), std::byte{0x01});
  fx.q8_scales.assign(static_cast<std::size_t>(*q8_s), std::byte{0});
  for (std::size_t i = 1; i < fx.q4_scales.size(); i += 2) {
    fx.q4_scales[i] = std::byte{0x04};
  }
  for (std::size_t i = 1; i < fx.q8_scales.size(); i += 2) {
    fx.q8_scales[i] = std::byte{0x04};
  }
  mutate(schema, fx);

  auto writer = ArtifactWriter::create(fx.path, schema);
  if (!writer) {
    fx.path.clear();
    return fx;
  }
  auto write = [&](std::string_view name, SpanKind kind,
                   std::vector<std::byte> const& bytes) {
    return writer->write_span(name, kind, bytes);
  };
  if (!write("lm_head.weight", SpanKind::Payload, fx.q8_payload) ||
      !write("model.embed_tokens.weight", SpanKind::Payload, fx.embed) ||
      !write("model.layers.0.mlp.down_proj.weight", SpanKind::Scales,
             fx.q4_scales) ||
      !write("model.layers.0.mlp.down_proj.weight", SpanKind::Payload,
             fx.q4_payload) ||
      !write("lm_head.weight", SpanKind::Scales, fx.q8_scales)) {
    fx.path.clear();
    return fx;
  }
  auto id = writer->finalize();
  if (!id) {
    fx.path.clear();
    return fx;
  }
  return fx;
}

inline Task003Fixture write_task003(std::filesystem::path dest) {
  return write_task003_mutated(
      std::move(dest), [](ArtifactSchema&, Task003Fixture&) {});
}

inline bool is_alias_id(ArtifactSchema const& schema, std::uint32_t id) {
  for (auto const& b : schema.shared_bindings) {
    if (b.alias_tensor_id == id) {
      return true;
    }
  }
  return false;
}

inline std::uint64_t tensor_record_bytes(TensorRecord const& t) {
  return 4 + 2 + t.logical_name.size() + 8 +
         16ull * t.shape.rank + 8 + 20 + 16 + 16;
}

inline std::uint64_t state_record_bytes(StateAllocation const& s) {
  return 48 + 8 + 16ull * s.shape_per_layer.rank;
}

inline std::vector<std::uint64_t> record_boundaries(
    ContainerHeader const& header, ArtifactSchema const& schema) {
  std::vector<std::uint64_t> bounds;
  auto add = [&](std::uint64_t off) { bounds.push_back(off); };
  add(0);
  add(8);
  add(10);
  add(12);
  add(14);
  add(16);
  add(24);
  add(32);
  add(64);
  for (auto const& t : schema.tensors) {
    if (is_alias_id(schema, t.tensor_id)) {
      continue;
    }
    if (!t.payload.empty()) {
      add(t.payload.offset);
      add(t.payload.offset + t.payload.length);
    }
    if (!t.scales.empty()) {
      add(t.scales.offset);
      add(t.scales.offset + t.scales.length);
    }
  }
  std::uint64_t const m = header.manifest_offset;
  add(m);
  std::uint64_t n = 0;
  auto rel = [&](std::uint64_t extra) {
    n += extra;
    add(m + n);
  };
  rel(2);
  rel(2 + schema.compiler.ident.size() + 16);
  rel(32);
  rel(32);
  rel(32);
  rel(4 + 4ull * schema.precision.bindings.size());
  rel(2);
  rel(4);
  for (auto const& t : schema.tensors) {
    rel(tensor_record_bytes(t));
  }
  rel(4);
  for (std::size_t i = 0; i < schema.shared_bindings.size(); ++i) {
    rel(12);
  }
  rel(4);
  for (std::size_t i = 0; i < schema.graph_bindings.size(); ++i) {
    rel(16);
  }
  rel(4);
  for (auto const& s : schema.state) {
    rel(state_record_bytes(s));
  }
  rel(4);
  for (std::size_t i = 0; i < schema.scratch.size(); ++i) {
    rel(16);
  }
  rel(4);
  for (std::size_t i = 0; i < schema.integrity.size(); ++i) {
    rel(kIntegrityRecordBytes);
  }
  std::sort(bounds.begin(), bounds.end());
  bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
  return bounds;
}

template <typename Mutator>
std::expected<std::vector<std::byte>, FormatError> mutate_schema(
    std::span<std::byte const> file, Mutator&& mutator) {
  auto header = decode_header(file.first(kHeaderSizeV0));
  if (!header) {
    return std::unexpected(header.error());
  }
  auto schema = decode_schema(file.subspan(
      static_cast<std::size_t>(header->manifest_offset),
      static_cast<std::size_t>(header->manifest_length)));
  if (!schema) {
    return std::unexpected(schema.error());
  }
  mutator(*schema);
  auto n = encoded_size(*schema);
  if (!n) {
    return std::unexpected(n.error());
  }
  header->manifest_length = static_cast<std::uint64_t>(*n);
  auto const end = checked_add(header->manifest_offset, header->manifest_length,
                               0, "manifest");
  if (!end) {
    return std::unexpected(end.error());
  }
  std::vector<std::byte> out(file.begin(), file.end());
  if (out.size() < *end) {
    out.resize(static_cast<std::size_t>(*end));
  } else {
    out.resize(static_cast<std::size_t>(*end));
  }
  std::array<std::byte, kHeaderSizeV0> header_bytes{};
  if (auto st = encode(*header, header_bytes); !st) {
    return std::unexpected(st.error());
  }
  std::copy(header_bytes.begin(), header_bytes.end(), out.begin());
  std::vector<std::byte> manifest(*n);
  if (auto st = encode(*schema, manifest); !st) {
    return std::unexpected(st.error());
  }
  std::copy(manifest.begin(), manifest.end(),
            out.begin() + static_cast<std::ptrdiff_t>(header->manifest_offset));
  return out;
}

}  // namespace qw38::format::test
