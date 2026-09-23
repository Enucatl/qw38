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
#include <tuple>
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
using qw38::format::SemanticNodeKind;
using qw38::format::SemanticScope;
using qw38::format::SharedBinding;
using qw38::format::SharedBindingRole;
using qw38::format::SpanKind;
using qw38::format::StateAllocation;
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
  if (layout == PhysicalLayoutId::CudaBf16TapMajorV0) {
    return LogicalPhysicalMapping{.kind = MappingKind::TapMajorConvC1T};
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
  return exp.node != SemanticNodeKind::LmHead ||
         exp.role == TensorRole::LmHeadWeight ||
         exp.role == TensorRole::FinalLanguageNorm ||
         exp.role == TensorRole::MtpNorm;
}

struct NormCounts {
  std::uint32_t additive{};
  std::uint32_t qk{};
  std::uint32_t gdn_gated{};
  std::uint32_t final_language{};
  std::uint32_t mtp{};
};

std::expected<void, CompilerError> validate_required_norm_bindings(
    std::span<GraphBinding const> bindings) {
  std::unordered_map<std::uint32_t, NormCounts> counts;
  for (auto const& binding : bindings) {
    auto& c = counts[binding.instance_id];
    switch (binding.role) {
      case TensorRole::AdditiveNorm:
        ++c.additive;
        break;
      case TensorRole::QkNorm:
        ++c.qk;
        break;
      case TensorRole::GdnGatedNorm:
        ++c.gdn_gated;
        break;
      case TensorRole::FinalLanguageNorm:
        ++c.final_language;
        break;
      case TensorRole::MtpNorm:
        ++c.mtp;
        break;
      default:
        break;
    }
  }

  auto require = [&](std::uint32_t instance, NormCounts expected,
                     std::string_view field)
      -> std::expected<void, CompilerError> {
    auto const got = counts[instance];
    if (got.additive != expected.additive || got.qk != expected.qk ||
        got.gdn_gated != expected.gdn_gated ||
        got.final_language != expected.final_language ||
        got.mtp != expected.mtp) {
      return std::unexpected(make_error(
          CompilerErrorCode::ArchitectureMismatch, field,
          "required semantic norm binding multiplicity is not satisfied"));
    }
    return {};
  };

  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    auto const mixer = mixer_instance(layer);
    if (is_full_attention_layer(layer)) {
      if (auto st = require(mixer, {.additive = 1, .qk = 2},
                            "gated_attention.norms");
          !st) {
        return st;
      }
    } else if (auto st = require(mixer, {.additive = 1, .gdn_gated = 1},
                                 "gated_delta_net.norms");
               !st) {
      return st;
    }
    if (auto st = require(mlp_instance(layer), {.additive = 1}, "mlp.norms");
        !st) {
      return st;
    }
  }
  if (auto st = require(kLmHeadInstance, {.final_language = 1},
                        "lm_head.norms");
      !st) {
    return st;
  }
  if (auto st = require(kMtpMixInstance, {.additive = 2}, "mtp_mix.norms");
      !st) {
    return st;
  }
  if (auto st = require(kMtpAttnInstance, {.additive = 1, .qk = 2},
                        "mtp_attention.norms");
      !st) {
    return st;
  }
  if (auto st = require(kMtpMlpInstance, {.additive = 1}, "mtp_mlp.norms");
      !st) {
    return st;
  }
  return require(kMtpLmHeadInstance, {.mtp = 1}, "mtp_lm_head.norms");
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

std::expected<void, CompilerError> verify_bf16_payload_impl(
    TensorRecord const& record, std::span<std::byte const> payload,
    std::span<std::byte const> source) {
  auto const& shape = record.shape;
  if (shape.rank == 0 || shape.rank > shape.logical.size()) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                      record.logical_name,
                                      "BF16 logical rank is invalid"));
  }
  std::uint64_t elements = 1;
  for (std::uint8_t i = 0; i < shape.rank; ++i) {
    auto const dim = shape.logical[i];
    if (dim == 0 || elements > std::numeric_limits<std::uint64_t>::max() / dim) {
      return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                        record.logical_name,
                                        "BF16 logical element count is invalid"));
    }
    elements *= dim;
  }
  if (elements > std::numeric_limits<std::size_t>::max() / 2 ||
      source.size() != elements * 2) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                      record.logical_name,
                                      "BF16 source size differs from geometry"));
  }
  if (payload.size() != source.size()) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                      record.logical_name,
                                      "BF16 payload size differs from source"));
  }
  if (record.mapping.kind == MappingKind::TapMajorConvC1T) {
    if (shape.rank != 3 || shape.logical[1] != 1) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                        record.logical_name,
                                        "tap-major mapping has invalid logical shape"));
    }
    auto const channels = shape.logical[0];
    auto const taps = shape.logical[2];
    for (std::uint64_t c = 0; c < channels; ++c) {
      for (std::uint64_t t = 0; t < taps; ++t) {
        auto const src_offset = (c * taps + t) * 2;
        auto const payload_offset = (t * channels + c) * 2;
        if (std::memcmp(source.data() + src_offset,
                        payload.data() + payload_offset, 2) != 0) {
          return std::unexpected(make_error(CompilerErrorCode::HashMismatch,
                                            record.logical_name,
                                            "reconstructed bytes differ from source"));
        }
      }
    }
    return {};
  }
  if (record.mapping.kind == MappingKind::DenseTileNK) {
    if (shape.rank != 2 || shape.logical[0] % kDenseTileRows != 0 ||
        shape.logical[1] % kDenseTileK != 0) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                        record.logical_name,
                                        "dense-tile mapping has invalid logical shape"));
    }
    auto const n = shape.logical[0];
    auto const k = shape.logical[1];
    auto const tiles_k = k / kDenseTileK;
    for (std::uint64_t tile_n = 0; tile_n < n / kDenseTileRows; ++tile_n) {
      for (std::uint64_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
        for (std::uint64_t row = 0; row < kDenseTileRows; ++row) {
          auto const logical_row = tile_n * kDenseTileRows + row;
          auto const logical_offset = (logical_row * k) + tile_k * kDenseTileK;
          auto const payload_offset =
              ((tile_n * tiles_k + tile_k) * kDenseTileRows + row) *
              kDenseTileK;
          auto const bytes = kDenseTileK * 2;
          if (std::memcmp(source.data() + logical_offset * 2,
                          payload.data() + payload_offset * 2,
                          static_cast<std::size_t>(bytes)) != 0) {
            return std::unexpected(make_error(CompilerErrorCode::HashMismatch,
                                              record.logical_name,
                                              "reconstructed bytes differ from source"));
          }
        }
      }
    }
    return {};
  }
  if (record.mapping.kind == MappingKind::Identity) {
    constexpr std::size_t kChunkBytes = 1u << 20;
    for (std::size_t offset = 0; offset < source.size(); offset += kChunkBytes) {
      auto const count = std::min(kChunkBytes, source.size() - offset);
      if (std::memcmp(source.data() + offset, payload.data() + offset, count) != 0) {
        return std::unexpected(make_error(CompilerErrorCode::HashMismatch,
                                          record.logical_name,
                                          "reconstructed bytes differ from source"));
      }
    }
    return {};
  }
  return std::unexpected(make_error(CompilerErrorCode::Internal,
                                    record.logical_name,
                                    "unsupported BF16 reconstruction mapping"));
}

}  // namespace

std::expected<void, CompilerError> verify_bf16_tensor(
    TensorRecord const& record, std::span<std::byte const> payload,
    std::span<std::byte const> source) {
  return verify_bf16_payload_impl(record, payload, source);
}

std::vector<StateAllocation> language_state_schema() {
  auto const schema = qw38::format::v0_language_state_schema();
  return {schema.begin(), schema.end()};
}

std::vector<ScratchAllocation> language_scratch_schema() {
  auto const schema = qw38::format::v0_language_scratch_schema();
  return {schema.begin(), schema.end()};
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

std::expected<void, CompilerError> verify_artifact_identities(
    ArtifactSchema const& artifact, CheckpointIdentities const& checkpoint,
    CompilerRevision const& revision) {
  auto mismatch = [](std::string_view field) {
    return std::unexpected(make_error(CompilerErrorCode::HashMismatch, field,
                                      "artifact identity does not match source"));
  };
  if (artifact.source_hash != checkpoint.source_hash) {
    return mismatch("source_hash");
  }
  if (artifact.config_hash != checkpoint.config_hash) {
    return mismatch("config_hash");
  }
  if (artifact.tokenizer_hash != checkpoint.tokenizer_hash) {
    return mismatch("tokenizer_hash");
  }
  if (artifact.compiler != revision) {
    return mismatch("compiler_revision");
  }
  return {};
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

  if (classified.included.size() == kIncludedTensors) {
    if (auto st = validate_required_norm_bindings(schema.graph_bindings); !st) {
      return std::unexpected(st.error());
    }
  }

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
  result.source_metadata_hash = ckpt->source_hash;
  result.config_hash = ckpt->config_hash;
  result.tokenizer_hash = ckpt->tokenizer_hash;
  result.format_policy = options.format_policy;
  result.language_instances =
      count_instances(schema->graph_bindings, SourceClass::Language);
  result.mtp_instances = count_instances(schema->graph_bindings,
                                         SourceClass::MtpRetainedDisabled);
  result.included_tensors =
      static_cast<std::uint32_t>(ckpt->classified.included.size());
  result.vision_excluded = ckpt->classified.vision_excluded;

  if (options.verify_reconstruction) {
    if (auto st = verify_compiled_artifact(output, checkpoint,
                                           options.format_policy,
                                           options.revision);
        !st) {
      return std::unexpected(st.error());
    }
    result.reconstruction_verified = true;
  }
  result.peak_rss_bytes = current_peak_rss_bytes();
  return result;
}

std::expected<CompileResult, CompilerError> compile_identity(
    std::filesystem::path const& checkpoint,
    std::filesystem::path const& output, CompileOptions const& options) {
  CompileOptions identity = options;
  identity.format_policy = WeightFormatPolicy::IdentityBf16;
  return compile_checkpoint(checkpoint, output, identity);
}

std::expected<void, CompilerError> verify_quantized_tensor(
    std::string_view name, LogicalQuantizerId quantizer,
    PhysicalLayoutId layout, std::uint64_t n, std::uint64_t k,
    std::span<std::byte const> source, std::span<std::byte const> payload,
    std::span<std::byte const> scales) {
  if (!qw38::format::quantizer_layout_pair_ok(quantizer, layout)) {
    return std::unexpected(make_error(
        CompilerErrorCode::Internal, name,
        "quantizer and physical layout disagree during verification"));
  }
  if (n == 0 || k == 0 || n % kDenseTileRows != 0 ||
      k % kDenseTileK != 0 ||
      n > std::numeric_limits<std::uint64_t>::max() / k) {
    return std::unexpected(make_error(
        CompilerErrorCode::ShapeMismatch, name,
        "quantized verification requires N%8==0 and K%256==0"));
  }
  auto const elements = n * k;
  auto const group = quantizer_group_size(quantizer);
  auto const packed_row =
      quantizer == LogicalQuantizerId::Q4G64V0
          ? qw38::format::kQ4PackedBytesPerTileRow
          : qw38::format::kQ8PackedBytesPerTileRow;
  auto const payload_bytes =
      quantizer == LogicalQuantizerId::Q4G64V0 ? elements / 2 : elements;
  if (group == 0 || elements > std::numeric_limits<std::uint64_t>::max() / 2 ||
      source.size() != elements * 2 ||
      payload.size() != payload_bytes ||
      scales.size() != elements / group * 2) {
    return std::unexpected(make_error(
        CompilerErrorCode::ShapeMismatch, name,
        "quantized verification span lengths do not match tensor geometry"));
  }

  constexpr std::size_t kMaxCodeTile =
      kDenseTileRows * qw38::format::kQ8PackedBytesPerTileRow;
  constexpr std::size_t kMaxScaleTile =
      kDenseTileRows * (kDenseTileK / qw38::format::kQ8GroupSize) * 2;
  std::array<float, kDenseTileK> row{};
  std::array<std::int8_t, qw38::format::kQ4GroupSize> group_codes{};
  std::array<std::byte, kMaxCodeTile> expected_codes{};
  std::array<std::byte, kMaxScaleTile> expected_scales{};

  auto const groups_per_tile = kDenseTileK / group;
  auto const code_tile_bytes = kDenseTileRows * packed_row;
  auto const scale_tile_bytes = kDenseTileRows * groups_per_tile * 2;
  auto const tiles_k = k / kDenseTileK;
  for (std::uint64_t tn = 0; tn < n / kDenseTileRows; ++tn) {
    for (std::uint64_t tk = 0; tk < tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const source_row = tn * kDenseTileRows + r;
        auto const source_col = tk * kDenseTileK;
        auto const source_offset = (source_row * k + source_col) * 2;
        for (std::uint32_t c = 0; c < kDenseTileK; ++c) {
          auto const bits =
              qw38::format::load_u16_le(source.data() + source_offset + c * 2);
          row[c] = qw38::format::bf16_to_fp32(bits);
        }
        for (std::uint32_t g = 0; g < groups_per_tile; ++g) {
          auto codes = std::span<std::int8_t>{group_codes}.first(group);
          auto scale = quantize_group_into(
              quantizer,
              std::span<float const>{row}.subspan(g * group, group), codes);
          if (!scale) {
            return std::unexpected(scale.error());
          }
          auto const scale_offset = (r * groups_per_tile + g) * 2;
          qw38::format::store_u16_le(expected_scales.data() + scale_offset,
                                      *scale);
          auto const code_offset = r * packed_row;
          if (quantizer == LogicalQuantizerId::Q4G64V0) {
            for (std::uint32_t i = 0; i < group; i += 2) {
              auto const lo = static_cast<std::uint8_t>(codes[i]) & 0x0Fu;
              auto const hi = static_cast<std::uint8_t>(codes[i + 1]) & 0x0Fu;
              expected_codes[code_offset + g * (group / 2) + i / 2] =
                  static_cast<std::byte>(lo | (hi << 4));
            }
          } else {
            for (std::uint32_t i = 0; i < group; ++i) {
              expected_codes[code_offset + g * group + i] =
                  static_cast<std::byte>(static_cast<std::uint8_t>(codes[i]));
            }
          }
        }
      }
      auto const tile = tn * tiles_k + tk;
      auto const got_codes =
          payload.subspan(tile * code_tile_bytes, code_tile_bytes);
      auto const got_scales =
          scales.subspan(tile * scale_tile_bytes, scale_tile_bytes);
      if (auto st = compare_bytes(
              got_codes,
              std::span<std::byte const>{expected_codes}.first(code_tile_bytes),
              name);
          !st) {
        return st;
      }
      if (auto st = compare_bytes(
              got_scales,
              std::span<std::byte const>{expected_scales}.first(scale_tile_bytes),
              name);
          !st) {
        return st;
      }
    }
  }
  return {};
}

std::expected<void, CompilerError> verify_compiled_artifact(
    std::filesystem::path const& artifact_path,
    std::filesystem::path const& checkpoint, WeightFormatPolicy policy,
    CompilerRevision const& revision) {
  if (auto st = verify_artifact_metadata(artifact_path, checkpoint, policy,
                                        revision);
      !st) {
    return st;
  }
  auto ckpt = open_checkpoint(checkpoint);
  if (!ckpt) {
    return std::unexpected(ckpt.error());
  }
  auto art = Artifact::open(artifact_path);
  if (!art) {
    return std::unexpected(from_format(art.error()));
  }
  CheckpointIdentities identities{
      .source_hash = ckpt->source_hash,
      .config_hash = ckpt->config_hash,
      .tokenizer_hash = ckpt->tokenizer_hash,
  };
  if (auto st = verify_artifact_identities(art->schema(), identities, revision);
      !st) {
    return st;
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
      auto const* record = art->find_tensor(item->expected.name);
      if (record == nullptr) {
        return std::unexpected(make_error(CompilerErrorCode::MissingTensor,
                                          item->expected.name,
                                          "artifact tensor metadata is absent"));
      }
      auto const& exp = item->expected;
      if (record->quantizer == LogicalQuantizerId::Q4G64V0 ||
          record->quantizer == LogicalQuantizerId::Q8G32V0) {
        auto scales = art->scales(exp.name);
        if (!scales) {
          return std::unexpected(from_format(scales.error()));
        }
        if (auto st = verify_quantized_tensor(
                exp.name, record->quantizer, record->layout,
                record->shape.logical[0], record->shape.logical[1], *src,
                *payload, *scales);
            !st) {
          return st;
        }
        continue;
      }
      if (auto st = verify_bf16_tensor(*record, *payload, *src); !st) {
        return st;
      }
    }
  }
  return {};
}

std::expected<void, CompilerError> verify_artifact_metadata(
    std::filesystem::path const& artifact_path,
    std::filesystem::path const& checkpoint, WeightFormatPolicy policy,
    CompilerRevision const& revision) {
  auto ckpt = open_checkpoint(checkpoint);
  if (!ckpt) {
    return std::unexpected(ckpt.error());
  }
  auto art = Artifact::open(artifact_path);
  if (!art) {
    return std::unexpected(from_format(art.error()));
  }
  auto expected = build_schema(ckpt->classified, ckpt->source_hash,
                               ckpt->config_hash, ckpt->tokenizer_hash,
                               revision, policy);
  if (!expected) {
    return std::unexpected(expected.error());
  }

  return compare_artifact_schema(art->schema(), *expected);
}

std::expected<void, CompilerError> compare_artifact_schema(
    ArtifactSchema const& actual, ArtifactSchema const& expected) {
  auto mismatch = [](std::string field, std::string detail) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                      std::move(field), std::move(detail)));
  };
  if (actual.manifest_version != expected.manifest_version) {
    return mismatch("manifest_version", "does not match requested schema");
  }
  if (actual.compiler != expected.compiler) {
    return mismatch("compiler", "does not match requested compiler revision");
  }
  if (actual.source_hash != expected.source_hash ||
      actual.config_hash != expected.config_hash ||
      actual.tokenizer_hash != expected.tokenizer_hash) {
    return mismatch("source_metadata", "checkpoint metadata identity differs");
  }
  auto got_precision = actual.precision;
  auto want_precision = expected.precision;
  auto by_precision_domain = [](auto const& a, auto const& b) {
    return std::tuple{a.domain, a.dtype} < std::tuple{b.domain, b.dtype};
  };
  std::sort(got_precision.bindings.begin(), got_precision.bindings.end(),
            by_precision_domain);
  std::sort(want_precision.bindings.begin(), want_precision.bindings.end(),
            by_precision_domain);
  auto got_state = actual.state;
  auto want_state = expected.state;
  auto by_state_kind = [](auto const& a, auto const& b) {
    return a.kind < b.kind;
  };
  std::sort(got_state.begin(), got_state.end(), by_state_kind);
  std::sort(want_state.begin(), want_state.end(), by_state_kind);
  auto got_scratch = actual.scratch;
  auto want_scratch = expected.scratch;
  auto by_scratch_kind = [](auto const& a, auto const& b) {
    return a.kind < b.kind;
  };
  std::sort(got_scratch.begin(), got_scratch.end(), by_scratch_kind);
  std::sort(want_scratch.begin(), want_scratch.end(), by_scratch_kind);
  if (got_precision != want_precision || actual.scope != expected.scope ||
      got_state != want_state || got_scratch != want_scratch) {
    return mismatch("policy_or_state_schema",
                    "precision, semantic scope, state, or scratch differs");
  }

  auto got_tensors = actual.tensors;
  auto want_tensors = expected.tensors;
  auto by_tensor_name = [](TensorRecord const& a, TensorRecord const& b) {
    return a.logical_name < b.logical_name;
  };
  std::sort(got_tensors.begin(), got_tensors.end(), by_tensor_name);
  std::sort(want_tensors.begin(), want_tensors.end(), by_tensor_name);
  if (got_tensors.size() != want_tensors.size()) {
    return mismatch("tensor_directory", "tensor membership count differs");
  }
  for (std::size_t i = 0; i < want_tensors.size(); ++i) {
    auto got = got_tensors[i];
    auto want = want_tensors[i];
    got.tensor_id = 0;
    want.tensor_id = 0;
    got.payload = {};
    got.scales = {};
    want.payload = {};
    want.scales = {};
    if (got != want) {
      return mismatch("tensor." + want.logical_name,
                      got.logical_name != want.logical_name
                          ? "tensor membership differs"
                          : "shape, storage, quantizer, layout, or mapping differs");
    }
  }

  auto got_graph = actual.graph_bindings;
  auto want_graph = expected.graph_bindings;
  auto name_for = [](ArtifactSchema const& schema, std::uint32_t id)
      -> std::string_view {
    for (auto const& tensor : schema.tensors) {
      if (tensor.tensor_id == id) return tensor.logical_name;
    }
    return {};
  };
  auto graph_key = [&](ArtifactSchema const& schema, GraphBinding const& b) {
    return std::tuple{b.instance_id, b.kind, b.role, b.layer_index,
                      name_for(schema, b.tensor_id)};
  };
  std::sort(got_graph.begin(), got_graph.end(), [&](auto const& a, auto const& b) {
    return graph_key(actual, a) < graph_key(actual, b);
  });
  std::sort(want_graph.begin(), want_graph.end(), [&](auto const& a, auto const& b) {
    return graph_key(expected, a) < graph_key(expected, b);
  });
  if (got_graph.size() != want_graph.size() ||
      !std::equal(got_graph.begin(), got_graph.end(), want_graph.begin(),
                  [&](auto const& a, auto const& b) {
                    return graph_key(actual, a) == graph_key(expected, b);
                  })) {
    return mismatch("graph_bindings", "semantic binding set differs");
  }

  auto got_shared = actual.shared_bindings;
  auto want_shared = expected.shared_bindings;
  auto shared_key = [&](ArtifactSchema const& schema, SharedBinding const& b) {
    return std::tuple{name_for(schema, b.owner_tensor_id),
                      name_for(schema, b.alias_tensor_id), b.role};
  };
  std::sort(got_shared.begin(), got_shared.end(), [&](auto const& a, auto const& b) {
    return shared_key(actual, a) < shared_key(actual, b);
  });
  std::sort(want_shared.begin(), want_shared.end(), [&](auto const& a, auto const& b) {
    return shared_key(expected, a) < shared_key(expected, b);
  });
  if (got_shared.size() != want_shared.size() ||
      !std::equal(got_shared.begin(), got_shared.end(), want_shared.begin(),
                  [&](auto const& a, auto const& b) {
                    return shared_key(actual, a) == shared_key(expected, b);
                  })) {
    return mismatch("shared_bindings", "canonical sharing differs");
  }
  return {};
}

std::expected<void, CompilerError> verify_identity_artifact(
    std::filesystem::path const& artifact_path,
    std::filesystem::path const& checkpoint,
    CompilerRevision const& revision) {
  return verify_compiled_artifact(artifact_path, checkpoint,
                                  WeightFormatPolicy::IdentityBf16, revision);
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
