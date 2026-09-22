#include "compiler/compile.hpp"

#include "compiler/quantization/quantizer.hpp"
#include "compiler/quantization/reference.hpp"
#include "compiler/transforms.hpp"
#include "format/floatcvt.hpp"
#include "format/pack.hpp"
#include "format/reader.hpp"
#include "format/schema.hpp"
#include "format/unpack.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace qw38::compiler {
namespace {

using qw38::format::ArithmeticDtype;
using qw38::format::Artifact;
using qw38::format::ArtifactSchema;
using qw38::format::ArtifactWriter;
using qw38::format::CompilerRevision;
using qw38::format::GraphBinding;
using qw38::format::Hash256;
using qw38::format::kDenseTileK;
using qw38::format::kDenseTileRows;
using qw38::format::kNoLayerIndex;
using qw38::format::LogicalPhysicalMapping;
using qw38::format::LogicalQuantizerId;
using qw38::format::MappingKind;
using qw38::format::PhysicalLayoutId;
using qw38::format::ScratchAllocation;
using qw38::format::ScratchKind;
using qw38::format::SemanticNodeKind;
using qw38::format::SemanticScope;
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

constexpr std::uint32_t kEmbedInstance = 0;
constexpr std::uint32_t kLmHeadInstance = 129;
constexpr std::uint32_t kMtpEmbedInstance = 130;
constexpr std::uint32_t kMtpMixInstance = 131;
constexpr std::uint32_t kMtpAttnInstance = 132;
constexpr std::uint32_t kMtpMlpInstance = 133;
constexpr std::uint32_t kMtpLmHeadInstance = 134;

std::uint32_t mixer_instance(std::uint32_t layer) { return 1 + layer * 2; }
std::uint32_t mlp_instance(std::uint32_t layer) { return 2 + layer * 2; }

TensorShape make_shape(std::initializer_list<std::uint64_t> dims) {
  TensorShape s{};
  s.rank = static_cast<std::uint8_t>(dims.size());
  std::uint8_t i = 0;
  for (auto d : dims) {
    s.logical[i] = d;
    s.padded[i] = d;
    ++i;
  }
  return s;
}

TensorShape artifact_shape(ExpectedTensor const& exp) {
  if (exp.layout == PhysicalLayoutId::CudaBf16TapMajorV0) {
    return make_shape({exp.shape.dims[2], exp.shape.dims[0]});
  }
  if (exp.shape.rank == 1) {
    return make_shape({exp.shape.dims[0]});
  }
  if (exp.shape.rank == 2) {
    return make_shape({exp.shape.dims[0], exp.shape.dims[1]});
  }
  return make_shape({exp.shape.dims[0], exp.shape.dims[1], exp.shape.dims[2]});
}

std::expected<std::uint64_t, CompilerError> expected_tensor_bytes(
    ExpectedTensor const& tensor) {
  if (tensor.shape.rank == 0 || tensor.shape.rank > tensor.shape.dims.size()) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                      tensor.name, "shape rank must be 1..3"));
  }
  std::uint64_t elements = 1;
  for (std::uint8_t i = 0; i < tensor.shape.rank; ++i) {
    auto const dim = tensor.shape.dims[i];
    if (dim == 0) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                        tensor.name,
                                        "shape extents must be nonzero"));
    }
    if (elements > std::numeric_limits<std::uint64_t>::max() / dim) {
      return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                        tensor.name,
                                        "shape element count overflows uint64"));
    }
    elements *= dim;
  }
  for (std::size_t i = tensor.shape.rank; i < tensor.shape.dims.size(); ++i) {
    if (tensor.shape.dims[i] != 0) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                        tensor.name,
                                        "unused shape dimensions must be zero"));
    }
  }
  if (elements > std::numeric_limits<std::uint64_t>::max() / 2) {
    return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                      tensor.name,
                                      "BF16 byte size overflows uint64"));
  }
  return elements * 2;
}

LogicalPhysicalMapping mapping_for(PhysicalLayoutId layout) {
  if (layout == PhysicalLayoutId::CudaQ4G64V0) {
    return LogicalPhysicalMapping{
        .kind = MappingKind::DenseTileNK,
        .tile_rows = kDenseTileRows,
        .tile_k = kDenseTileK,
        .group_size = qw38::format::kQ4GroupSize,
        .packed_bytes_per_tile_row = qw38::format::kQ4PackedBytesPerTileRow,
    };
  }
  if (layout == PhysicalLayoutId::CudaQ8G32V0) {
    return LogicalPhysicalMapping{
        .kind = MappingKind::DenseTileNK,
        .tile_rows = kDenseTileRows,
        .tile_k = kDenseTileK,
        .group_size = qw38::format::kQ8GroupSize,
        .packed_bytes_per_tile_row = qw38::format::kQ8PackedBytesPerTileRow,
    };
  }
  if (layout == PhysicalLayoutId::CudaBf16DenseTileV0) {
    return LogicalPhysicalMapping{
        .kind = MappingKind::DenseTileNK,
        .tile_rows = kDenseTileRows,
        .tile_k = kDenseTileK,
        .group_size = 0,
        .packed_bytes_per_tile_row = qw38::format::kBf16PackedBytesPerTileRow,
    };
  }
  return LogicalPhysicalMapping{.kind = MappingKind::Identity};
}

std::uint32_t instance_for(ExpectedTensor const& exp) {
  if (exp.source_class == SourceClass::MtpRetainedDisabled) {
    switch (exp.node) {
      case SemanticNodeKind::MtpMix:
        return kMtpMixInstance;
      case SemanticNodeKind::GatedAttention:
        return kMtpAttnInstance;
      case SemanticNodeKind::Mlp:
        return kMtpMlpInstance;
      case SemanticNodeKind::LmHead:
        return kMtpLmHeadInstance;
      case SemanticNodeKind::Embed:
        return kMtpEmbedInstance;
      default:
        return kMtpMixInstance;
    }
  }
  switch (exp.node) {
    case SemanticNodeKind::Embed:
      return kEmbedInstance;
    case SemanticNodeKind::LmHead:
      return kLmHeadInstance;
    case SemanticNodeKind::Mlp:
      return mlp_instance(exp.layer_index);
    case SemanticNodeKind::GatedAttention:
    case SemanticNodeKind::GatedDeltaNet:
      return mixer_instance(exp.layer_index);
    default:
      return kEmbedInstance;
  }
}

bool bindable(ExpectedTensor const& exp) {
  if (exp.node == SemanticNodeKind::LmHead &&
      exp.role != TensorRole::LmHeadWeight) {
    return false;
  }
  return true;
}

std::expected<void, CompilerError> write_span(
    ArtifactWriter& writer, std::string_view name,
    std::span<std::byte const> bytes) {
  auto st = writer.write_span(name, SpanKind::Payload, bytes);
  if (!st) {
    return std::unexpected(from_format(st.error()));
  }
  return {};
}

std::expected<void, CompilerError> emit_tiled(
    ArtifactWriter& writer, std::string_view name,
    std::span<std::byte const> src, std::uint64_t n, std::uint64_t k) {
  auto extents = dense_tile_extents(n, k);
  if (!extents) {
    return std::unexpected(extents.error());
  }
  if (src.size() != n * k * 2) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, name,
                                      "dense source byte length"));
  }
  std::array<std::byte, kDenseTileRows * kDenseTileK * 2> tile{};
  for (std::uint64_t tn = 0; tn < extents->tiles_n; ++tn) {
    for (std::uint64_t tk = 0; tk < extents->tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const row = tn * kDenseTileRows + r;
        auto const src_off = (row * k + tk * kDenseTileK) * 2;
        auto const dst_off = r * kDenseTileK * 2;
        std::memcpy(tile.data() + dst_off, src.data() + src_off,
                    kDenseTileK * 2);
        for (std::uint32_t c = 0; c < kDenseTileK; ++c) {
          std::uint16_t bits = 0;
          std::memcpy(&bits, tile.data() + dst_off + c * 2, 2);
          if (!bf16_is_finite(bits)) {
            return std::unexpected(make_error(CompilerErrorCode::Nonfinite, name,
                                              "nonfinite BF16 dense weight"));
          }
        }
      }
      if (auto st = write_span(writer, name, tile); !st) {
        return st;
      }
    }
  }
  return {};
}

std::expected<void, CompilerError> emit_quantized(
    ArtifactWriter& writer, std::string_view name,
    std::span<std::byte const> src, std::uint64_t n, std::uint64_t k,
    LogicalQuantizerId quantizer, PhysicalLayoutId layout) {
  if (n == 0 || k == 0 || n % kDenseTileRows != 0 || k % kDenseTileK != 0) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, name,
                                      "quantized dense N must divide 8 and K 256"));
  }
  if (src.size() != n * k * 2) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, name,
                                      "dense source byte length"));
  }
  auto const group = quantizer_group_size(quantizer);
  if (group == 0 || k % group != 0) {
    return std::unexpected(make_error(CompilerErrorCode::Internal, name,
                                      "quantizer group does not divide K"));
  }
  std::vector<std::byte> scale_bytes;
  std::vector<float> row(static_cast<std::size_t>(k));
  for (std::uint64_t tn = 0; tn < n / kDenseTileRows; ++tn) {
    qw38::format::LogicalWeightCodes slice;
    slice.quantizer = quantizer;
    slice.n = kDenseTileRows;
    slice.k = k;
    slice.group_size = group;
    slice.qmax = quantizer_qmax(quantizer);
    slice.codes.resize(static_cast<std::size_t>(kDenseTileRows * k));
    slice.scales.resize(static_cast<std::size_t>(kDenseTileRows * (k / group)));
    for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
      auto const src_row = src.subspan(((tn * kDenseTileRows + r) * k) * 2, k * 2);
      for (std::uint64_t col = 0; col < k; ++col) {
        auto const bits = qw38::format::load_u16_le(src_row.data() + col * 2);
        row[static_cast<std::size_t>(col)] = qw38::format::bf16_to_fp32(bits);
      }
      for (std::uint64_t g = 0; g < k / group; ++g) {
        auto qg = quantize_group(
            quantizer, std::span<float const>{row.data() + g * group, group});
        if (!qg) {
          return std::unexpected(qg.error());
        }
        slice.scales[static_cast<std::size_t>(r * (k / group) + g)] =
            qg->scale_bits;
        std::copy(qg->codes.begin(), qg->codes.end(),
                  slice.codes.begin() +
                      static_cast<std::ptrdiff_t>(r * k + g * group));
      }
    }
    auto packed = qw38::format::pack_cuda_v0(quantizer, layout, slice);
    if (!packed) {
      return std::unexpected(from_format(packed.error()));
    }
    if (auto st = write_span(writer, name, packed->codes); !st) {
      return st;
    }
    scale_bytes.insert(scale_bytes.end(), packed->scales.begin(),
                       packed->scales.end());
  }
  auto st = writer.write_span(name, SpanKind::Scales, scale_bytes);
  if (!st) {
    return std::unexpected(from_format(st.error()));
  }
  return {};
}

std::expected<void, CompilerError> emit_identity_bytes(
    ArtifactWriter& writer, std::string_view name,
    std::span<std::byte const> src) {
  if (auto st = require_all_finite_bf16(src, name); !st) {
    return st;
  }
  constexpr std::size_t kChunk = 1 << 20;
  for (std::size_t off = 0; off < src.size(); off += kChunk) {
    auto const n = std::min(kChunk, src.size() - off);
    if (auto st = write_span(writer, name, src.subspan(off, n)); !st) {
      return st;
    }
  }
  return {};
}

std::expected<void, CompilerError> compare_bytes(std::span<std::byte const> a,
                                                 std::span<std::byte const> b,
                                                 std::string_view name) {
  if (a.size() != b.size() ||
      (a.size() > 0 && std::memcmp(a.data(), b.data(), a.size()) != 0)) {
    return std::unexpected(make_error(CompilerErrorCode::HashMismatch, name,
                                      "reconstructed bytes differ from source"));
  }
  return {};
}

std::expected<void, CompilerError> compare_tiled(
    std::span<std::byte const> tiled, std::span<std::byte const> src,
    std::uint64_t n, std::uint64_t k, std::string_view name) {
  auto extents = dense_tile_extents(n, k);
  if (!extents) {
    return std::unexpected(extents.error());
  }
  if (src.size() != n * k * 2 || tiled.size() != src.size()) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, name,
                                      "tiled reconstruction size"));
  }
  std::array<std::byte, kDenseTileRows * kDenseTileK * 2> tile{};
  std::size_t cursor = 0;
  for (std::uint64_t tn = 0; tn < extents->tiles_n; ++tn) {
    for (std::uint64_t tk = 0; tk < extents->tiles_k; ++tk) {
      std::memcpy(tile.data(), tiled.data() + cursor, tile.size());
      cursor += tile.size();
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const row = tn * kDenseTileRows + r;
        auto const src_off = (row * k + tk * kDenseTileK) * 2;
        if (std::memcmp(tile.data() + r * kDenseTileK * 2, src.data() + src_off,
                        kDenseTileK * 2) != 0) {
          return std::unexpected(make_error(CompilerErrorCode::HashMismatch, name,
                                            "dense tile reconstruction mismatch"));
        }
      }
    }
  }
  return {};
}

}  // namespace

std::vector<StateAllocation> language_state_schema() {
  StateAllocation gdn{};
  gdn.kind = StateKind::GdnS;
  gdn.dtype = ArithmeticDtype::Fp32;
  gdn.layout = PhysicalLayoutId::CudaFp32GdnSHvKV0;
  gdn.shape_per_layer = make_shape({48, 128, 128});
  gdn.layer_count = 48;
  gdn.component_count = 1;
  gdn.bytes_per_layer = 3145728;
  gdn.total_bytes = 150994944;

  StateAllocation conv{};
  conv.kind = StateKind::ConvolutionHistory;
  conv.dtype = ArithmeticDtype::Bf16;
  conv.layout = PhysicalLayoutId::CudaBf16ConvHistoryV0;
  conv.shape_per_layer = make_shape({3, 10240});
  conv.layer_count = 48;
  conv.component_count = 1;
  conv.bytes_per_layer = 61440;
  conv.total_bytes = 2949120;

  StateAllocation kv{};
  kv.kind = StateKind::KvCache;
  kv.dtype = ArithmeticDtype::Bf16;
  kv.layout = PhysicalLayoutId::CudaBf16KvCacheV0;
  kv.shape_per_layer = make_shape({4, 256});
  kv.layer_count = 16;
  kv.component_count = 2;
  kv.declared_capacity = 0;
  kv.bytes_per_token = 65536;
  kv.bytes_per_layer = 0;
  kv.total_bytes = 0;
  kv.populated_length_distinct_from_capacity = true;
  return {gdn, conv, kv};
}

std::vector<ScratchAllocation> language_scratch_schema() {
  return {
      ScratchAllocation{.kind = ScratchKind::ResidualH,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 20480},
      ScratchAllocation{.kind = ScratchKind::ResidualHMid,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 20480},
      ScratchAllocation{.kind = ScratchKind::NormalizedHidden,
                        .dtype = ArithmeticDtype::Bf16,
                        .bytes = 10240},
      ScratchAllocation{.kind = ScratchKind::GdnWorkspace,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 107264},
      ScratchAllocation{.kind = ScratchKind::AttentionWorkspace,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 78016},
      ScratchAllocation{.kind = ScratchKind::MlpSwiglu,
                        .dtype = ArithmeticDtype::Bf16,
                        .bytes = 34816},
      ScratchAllocation{.kind = ScratchKind::Logits,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 993280},
  };
}

std::uint32_t count_instances(std::span<GraphBinding const> bindings,
                              SourceClass which) noexcept {
  std::unordered_set<std::uint32_t> ids;
  for (auto const& b : bindings) {
    bool const mtp = b.instance_id >= kMtpEmbedInstance;
    if ((which == SourceClass::MtpRetainedDisabled) != mtp) {
      continue;
    }
    ids.insert(b.instance_id);
  }
  return static_cast<std::uint32_t>(ids.size());
}

std::uint64_t current_peak_rss_bytes() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      std::uint64_t kb = 0;
      std::sscanf(line.c_str(), "VmHWM: %lu", &kb);
      return kb * 1024;
    }
  }
  return 0;
}

std::expected<ArtifactSchema, CompilerError> build_schema(
    ClassifiedCheckpoint const& classified, Hash256 const& source_hash,
    Hash256 const& config_hash, Hash256 const& tokenizer_hash,
    CompilerRevision const& revision, WeightFormatPolicy policy) {
  ArtifactSchema schema{};
  schema.compiler = revision;
  schema.source_hash = source_hash;
  schema.config_hash = config_hash;
  schema.tokenizer_hash = tokenizer_hash;
  schema.precision = v0_precision_policy();
  schema.scope = SemanticScope::LanguagePlusMtpDescriptors;
  schema.state = language_state_schema();
  schema.scratch = language_scratch_schema();

  schema.tensors.reserve(classified.included.size() + 3);
  schema.graph_bindings.reserve(classified.included.size() + 32);

  std::uint32_t embed_id = 0;
  std::uint32_t lm_head_id = 0;
  std::uint32_t next_id = 1;
  for (auto const& item : classified.included) {
    auto const source_bytes = expected_tensor_bytes(item.expected);
    if (!source_bytes) {
      return std::unexpected(source_bytes.error());
    }
    TensorRecord rec{};
    rec.tensor_id = next_id++;
    rec.logical_name = item.expected.name;
    rec.shape = artifact_shape(item.expected);
    auto const fmt = select_weight_format(item.expected.family,
                                          item.expected.layout, policy);
    rec.storage = fmt.storage;
    rec.quantizer = fmt.quantizer;
    rec.layout = fmt.layout;
    rec.mapping = mapping_for(fmt.layout);
    if (item.expected.family == TensorFamily::Embed) {
      embed_id = rec.tensor_id;
    }
    if (item.expected.family == TensorFamily::LmHead) {
      lm_head_id = rec.tensor_id;
    }
    if (bindable(item.expected)) {
      schema.graph_bindings.push_back(GraphBinding{
          .instance_id = instance_for(item.expected),
          .kind = item.expected.node,
          .role = item.expected.role,
          .layer_index = item.expected.layer_index,
          .tensor_id = rec.tensor_id,
      });
    }
    schema.tensors.push_back(std::move(rec));
  }

  TensorRecord rope{};
  rope.tensor_id = next_id++;
  rope.logical_name = kRopeInvFreqName;
  rope.shape = make_shape({kRopeFreqs});
  rope.storage = StorageClass::Fp32;
  rope.quantizer = LogicalQuantizerId::None;
  rope.layout = PhysicalLayoutId::CudaFp32VectorV0;
  rope.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};
  auto const rope_id = rope.tensor_id;
  schema.tensors.push_back(std::move(rope));
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    if (!is_full_attention_layer(layer)) {
      continue;
    }
    schema.graph_bindings.push_back(GraphBinding{
        .instance_id = mixer_instance(layer),
        .kind = SemanticNodeKind::GatedAttention,
        .role = TensorRole::VectorWeight,
        .layer_index = layer,
        .tensor_id = rope_id,
    });
  }
  schema.graph_bindings.push_back(GraphBinding{
      .instance_id = kMtpAttnInstance,
      .kind = SemanticNodeKind::GatedAttention,
      .role = TensorRole::VectorWeight,
      .layer_index = 0,
      .tensor_id = rope_id,
  });

  if (embed_id == 0 || lm_head_id == 0) {
    return std::unexpected(make_error(CompilerErrorCode::MissingTensor, "embed",
                                      "identity compile requires embed and lm_head"));
  }

  TensorRecord const* embed_rec = nullptr;
  TensorRecord const* head_rec = nullptr;
  for (auto const& t : schema.tensors) {
    if (t.tensor_id == embed_id) {
      embed_rec = &t;
    }
    if (t.tensor_id == lm_head_id) {
      head_rec = &t;
    }
  }

  TensorRecord mtp_embed = *embed_rec;
  mtp_embed.tensor_id = next_id++;
  mtp_embed.logical_name = kMtpEmbedAliasName;
  auto const mtp_embed_id = mtp_embed.tensor_id;
  schema.tensors.push_back(std::move(mtp_embed));

  TensorRecord mtp_head = *head_rec;
  mtp_head.tensor_id = next_id++;
  mtp_head.logical_name = kMtpLmHeadAliasName;
  auto const mtp_head_id = mtp_head.tensor_id;
  schema.tensors.push_back(std::move(mtp_head));

  schema.graph_bindings.push_back(GraphBinding{
      .instance_id = kMtpEmbedInstance,
      .kind = SemanticNodeKind::Embed,
      .role = TensorRole::EmbeddingTable,
      .layer_index = kNoLayerIndex,
      .tensor_id = mtp_embed_id,
  });
  schema.graph_bindings.push_back(GraphBinding{
      .instance_id = kMtpLmHeadInstance,
      .kind = SemanticNodeKind::LmHead,
      .role = TensorRole::LmHeadWeight,
      .layer_index = kNoLayerIndex,
      .tensor_id = mtp_head_id,
  });
  schema.shared_bindings.push_back(SharedBinding{
      .owner_tensor_id = embed_id,
      .alias_tensor_id = mtp_embed_id,
      .role = SharedBindingRole::MtpEmbeddingAlias,
  });
  schema.shared_bindings.push_back(SharedBinding{
      .owner_tensor_id = lm_head_id,
      .alias_tensor_id = mtp_head_id,
      .role = SharedBindingRole::MtpLmHeadAlias,
  });

  return schema;
}

std::expected<ArtifactSchema, CompilerError> build_identity_schema(
    ClassifiedCheckpoint const& classified, Hash256 const& source_hash,
    Hash256 const& config_hash, Hash256 const& tokenizer_hash,
    CompilerRevision const& revision) {
  return build_schema(classified, source_hash, config_hash, tokenizer_hash,
                      revision, WeightFormatPolicy::IdentityBf16);
}

std::expected<void, CompilerError> emit_classified_tensor(
    ArtifactWriter& writer, ClassifiedTensor const& item,
    std::span<std::byte const> src, WeightFormatPolicy policy) {
  auto const& exp = item.expected;
  auto const fmt = select_weight_format(exp.family, exp.layout, policy);
  if (fmt.quantizer == LogicalQuantizerId::Q4G64V0 ||
      fmt.quantizer == LogicalQuantizerId::Q8G32V0) {
    return emit_quantized(writer, exp.name, src, exp.shape.dims[0],
                          exp.shape.dims[1], fmt.quantizer, fmt.layout);
  }
  if (fmt.layout == PhysicalLayoutId::CudaBf16DenseTileV0) {
    return emit_tiled(writer, exp.name, src, exp.shape.dims[0],
                      exp.shape.dims[1]);
  }
  if (fmt.layout == PhysicalLayoutId::CudaBf16TapMajorV0) {
    auto packed = conv_to_tap_major(src, exp.shape.dims[0], exp.shape.dims[2]);
    if (!packed) {
      return std::unexpected(packed.error());
    }
    if (auto st = require_all_finite_bf16(src, exp.name); !st) {
      return st;
    }
    return write_span(writer, exp.name, *packed);
  }
  return emit_identity_bytes(writer, exp.name, src);
}

std::expected<CompileResult, CompilerError> compile_checkpoint(
    std::filesystem::path const& checkpoint,
    std::filesystem::path const& output, CompileOptions const& options) {
  auto ckpt = open_checkpoint(checkpoint);
  if (!ckpt) {
    return std::unexpected(ckpt.error());
  }
  auto schema = build_schema(ckpt->classified, ckpt->source_hash,
                             ckpt->config_hash, ckpt->tokenizer_hash,
                             options.revision, options.format_policy);
  if (!schema) {
    return std::unexpected(schema.error());
  }

  auto writer = ArtifactWriter::create(output, *schema);
  if (!writer) {
    return std::unexpected(from_format(writer.error()));
  }

  auto rope = generate_rope_inv_freq();
  if (auto st = write_span(*writer, kRopeInvFreqName, rope); !st) {
    return std::unexpected(st.error());
  }

  std::unordered_map<std::string, std::vector<ClassifiedTensor const*>> by_shard;
  for (auto const& item : ckpt->classified.included) {
    by_shard[item.source.shard].push_back(&item);
  }
  for (auto const& [shard_name, items] : by_shard) {
    auto it = ckpt->shards.find(shard_name);
    if (it == ckpt->shards.end()) {
      return std::unexpected(make_error(CompilerErrorCode::MissingTensor,
                                        shard_name, "shard path missing"));
    }
    auto mapped = MappedShard::open(it->second.path);
    if (!mapped) {
      return std::unexpected(mapped.error());
    }
    for (auto const* item : items) {
      auto src = mapped->tensor_bytes(item->source);
      if (!src) {
        return std::unexpected(src.error());
      }
      if (auto st = emit_classified_tensor(*writer, *item, *src,
                                           options.format_policy);
          !st) {
        return std::unexpected(st.error());
      }
    }
  }

  auto identity = writer->finalize();
  if (!identity) {
    return std::unexpected(from_format(identity.error()));
  }

  CompileResult result{};
  result.identity = *identity;
  result.language_instances =
      count_instances(schema->graph_bindings, SourceClass::Language);
  result.mtp_instances = count_instances(schema->graph_bindings,
                                         SourceClass::MtpRetainedDisabled);
  result.included_tensors =
      static_cast<std::uint32_t>(ckpt->classified.included.size());
  result.vision_excluded = ckpt->classified.vision_excluded;
  result.peak_rss_bytes = current_peak_rss_bytes();

  if (options.verify_reconstruction) {
    if (auto st = verify_compiled_artifact(output, checkpoint,
                                           options.format_policy);
        !st) {
      return std::unexpected(st.error());
    }
  }
  return result;
}

std::expected<CompileResult, CompilerError> compile_identity(
    std::filesystem::path const& checkpoint,
    std::filesystem::path const& output, CompileOptions const& options) {
  CompileOptions identity = options;
  identity.format_policy = WeightFormatPolicy::IdentityBf16;
  return compile_checkpoint(checkpoint, output, identity);
}

std::expected<void, CompilerError> verify_compiled_artifact(
    std::filesystem::path const& artifact_path,
    std::filesystem::path const& checkpoint, WeightFormatPolicy policy) {
  auto ckpt = open_checkpoint(checkpoint);
  if (!ckpt) {
    return std::unexpected(ckpt.error());
  }
  auto art = Artifact::open(artifact_path);
  if (!art) {
    return std::unexpected(from_format(art.error()));
  }
  auto rope_got = art->payload(kRopeInvFreqName);
  if (!rope_got) {
    return std::unexpected(from_format(rope_got.error()));
  }
  auto rope_want = generate_rope_inv_freq();
  if (auto st = compare_bytes(*rope_got, rope_want, kRopeInvFreqName); !st) {
    return st;
  }

  std::unordered_map<std::string, std::vector<ClassifiedTensor const*>> by_shard;
  for (auto const& item : ckpt->classified.included) {
    by_shard[item.source.shard].push_back(&item);
  }
  for (auto const& [shard_name, items] : by_shard) {
    auto it = ckpt->shards.find(shard_name);
    if (it == ckpt->shards.end()) {
      return std::unexpected(make_error(CompilerErrorCode::MissingTensor,
                                        shard_name, "shard path missing"));
    }
    auto mapped = MappedShard::open(it->second.path);
    if (!mapped) {
      return std::unexpected(mapped.error());
    }
    for (auto const* item : items) {
      auto src = mapped->tensor_bytes(item->source);
      if (!src) {
        return std::unexpected(src.error());
      }
      auto payload = art->payload(item->expected.name);
      if (!payload) {
        return std::unexpected(from_format(payload.error()));
      }
      auto const& exp = item->expected;
      auto const fmt = select_weight_format(exp.family, exp.layout, policy);
      if (fmt.quantizer == LogicalQuantizerId::Q4G64V0 ||
          fmt.quantizer == LogicalQuantizerId::Q8G32V0) {
        auto scales = art->scales(exp.name);
        if (!scales) {
          return std::unexpected(from_format(scales.error()));
        }
        auto want = quantize_bf16(fmt.quantizer, exp.shape.dims[0],
                                  exp.shape.dims[1], *src);
        if (!want) {
          return std::unexpected(want.error());
        }
        auto packed = qw38::format::pack_cuda_v0(fmt.quantizer, fmt.layout, *want);
        if (!packed) {
          return std::unexpected(from_format(packed.error()));
        }
        if (auto st = compare_bytes(*payload, packed->codes, exp.name); !st) {
          return st;
        }
        if (auto st = compare_bytes(*scales, packed->scales, exp.name); !st) {
          return st;
        }
        continue;
      }
      if (fmt.layout == PhysicalLayoutId::CudaBf16DenseTileV0) {
        if (auto st = compare_tiled(*payload, *src, exp.shape.dims[0],
                                    exp.shape.dims[1], exp.name);
            !st) {
          return st;
        }
      } else if (fmt.layout == PhysicalLayoutId::CudaBf16TapMajorV0) {
        auto restored =
            conv_from_tap_major(*payload, exp.shape.dims[0], exp.shape.dims[2]);
        if (!restored) {
          return std::unexpected(restored.error());
        }
        if (auto st = compare_bytes(*restored, *src, exp.name); !st) {
          return st;
        }
      } else {
        if (auto st = compare_bytes(*payload, *src, exp.name); !st) {
          return st;
        }
      }
    }
  }
  return {};
}

std::expected<void, CompilerError> verify_identity_artifact(
    std::filesystem::path const& artifact_path,
    std::filesystem::path const& checkpoint) {
  return verify_compiled_artifact(artifact_path, checkpoint,
                                  WeightFormatPolicy::IdentityBf16);
}

std::expected<CompileResult, CompilerError> compile_synthetic(
    std::filesystem::path const& output, Hash256 const& source_hash,
    Hash256 const& config_hash, Hash256 const& tokenizer_hash,
    CompilerRevision const& revision, std::vector<SyntheticTensor> tensors,
    WeightFormatPolicy policy) {
  ClassifiedCheckpoint classified{};
  classified.included.reserve(tensors.size());
  for (auto& t : tensors) {
    auto expected_bytes = expected_tensor_bytes(t.expected);
    if (!expected_bytes) {
      return std::unexpected(expected_bytes.error());
    }
    if (t.bytes.size() != *expected_bytes) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                        t.expected.name,
                                        "synthetic BF16 byte length mismatch"));
    }
    SourceTensor src{};
    src.name = t.expected.name;
    src.dtype = "BF16";
    src.shape.assign(t.expected.shape.dims.begin(),
                     t.expected.shape.dims.begin() + t.expected.shape.rank);
    src.nbytes = t.bytes.size();
    classified.included.push_back(ClassifiedTensor{
        .expected = std::move(t.expected),
        .source = std::move(src),
    });
  }
  auto schema = build_schema(classified, source_hash, config_hash,
                             tokenizer_hash, revision, policy);
  if (!schema) {
    return std::unexpected(schema.error());
  }
  auto writer = ArtifactWriter::create(output, *schema);
  if (!writer) {
    return std::unexpected(from_format(writer.error()));
  }
  auto rope = generate_rope_inv_freq();
  if (auto st = write_span(*writer, kRopeInvFreqName, rope); !st) {
    return std::unexpected(st.error());
  }
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    ClassifiedTensor item{};
    item.expected = classified.included[i].expected;
    if (auto st = emit_classified_tensor(*writer, item, tensors[i].bytes, policy);
        !st) {
      return std::unexpected(st.error());
    }
  }
  auto identity = writer->finalize();
  if (!identity) {
    return std::unexpected(from_format(identity.error()));
  }
  CompileResult result{};
  result.identity = *identity;
  result.language_instances =
      count_instances(schema->graph_bindings, SourceClass::Language);
  result.mtp_instances = count_instances(schema->graph_bindings,
                                         SourceClass::MtpRetainedDisabled);
  result.included_tensors = static_cast<std::uint32_t>(tensors.size());
  result.peak_rss_bytes = current_peak_rss_bytes();
  return result;
}

}  // namespace qw38::compiler
