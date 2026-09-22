#include "compiler/identity.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace qw38::compiler {
namespace {

using qw38::format::kNoLayerIndex;
using qw38::format::PhysicalLayoutId;
using qw38::format::SemanticNodeKind;
using qw38::format::TensorRole;

constexpr TensorShapeSpec shape1(std::uint64_t a) {
  return TensorShapeSpec{.rank = 1, .dims = {a, 0, 0}};
}
constexpr TensorShapeSpec shape2(std::uint64_t a, std::uint64_t b) {
  return TensorShapeSpec{.rank = 2, .dims = {a, b, 0}};
}
constexpr TensorShapeSpec shape3(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
  return TensorShapeSpec{.rank = 3, .dims = {a, b, c}};
}

constexpr char const kLayerPrefix[] = "model.language_model.layers.";

constexpr FamilyIdentity kTable[] = {
    {TensorFamily::Embed, SourceClass::Language,
     "model.language_model.embed_tokens.weight", "", false, false, false,
     shape2(kVocab, kHidden), PhysicalLayoutId::CudaBf16RowMajorV0,
     TensorRole::EmbeddingTable, SemanticNodeKind::Embed},
    {TensorFamily::FinalNorm, SourceClass::Language,
     "model.language_model.norm.weight", "", false, false, false, shape1(kHidden),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::FinalLanguageNorm,
     SemanticNodeKind::LmHead},
    {TensorFamily::LmHead, SourceClass::Language, "lm_head.weight", "", false,
     false, false, shape2(kVocab, kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::LmHeadWeight,
     SemanticNodeKind::LmHead},
    {TensorFamily::InputLayernorm, SourceClass::Language, kLayerPrefix,
     ".input_layernorm.weight", true, false, false, shape1(kHidden),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::AdditiveNorm,
     SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::PostAttentionLayernorm, SourceClass::Language, kLayerPrefix,
     ".post_attention_layernorm.weight", true, false, false, shape1(kHidden),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::AdditiveNorm,
     SemanticNodeKind::Mlp},
    {TensorFamily::LinearAttnALog, SourceClass::Language, kLayerPrefix,
     ".linear_attn.A_log", true, true, false, shape1(kLinearValueHeads),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::TimeParameter,
     SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::LinearAttnConv1d, SourceClass::Language, kLayerPrefix,
     ".linear_attn.conv1d.weight", true, true, false,
     shape3(kQkvWidth, 1, kConvKernel), PhysicalLayoutId::CudaBf16TapMajorV0,
     TensorRole::ConvWeight, SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::LinearAttnDtBias, SourceClass::Language, kLayerPrefix,
     ".linear_attn.dt_bias", true, true, false, shape1(kLinearValueHeads),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::TimeParameter,
     SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::LinearAttnInProjA, SourceClass::Language, kLayerPrefix,
     ".linear_attn.in_proj_a.weight", true, true, false,
     shape2(kLinearValueHeads, kHidden), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::LinearAttnInProjB, SourceClass::Language, kLayerPrefix,
     ".linear_attn.in_proj_b.weight", true, true, false,
     shape2(kLinearValueHeads, kHidden), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::LinearAttnInProjQkv, SourceClass::Language, kLayerPrefix,
     ".linear_attn.in_proj_qkv.weight", true, true, false,
     shape2(kQkvWidth, kHidden), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::LinearAttnInProjZ, SourceClass::Language, kLayerPrefix,
     ".linear_attn.in_proj_z.weight", true, true, false,
     shape2(kZWidth, kHidden), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::LinearAttnNorm, SourceClass::Language, kLayerPrefix,
     ".linear_attn.norm.weight", true, true, false, shape1(kLinearHeadDim),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::GdnGatedNorm,
     SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::LinearAttnOutProj, SourceClass::Language, kLayerPrefix,
     ".linear_attn.out_proj.weight", true, true, false,
     shape2(kHidden, kZWidth), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::GatedDeltaNet},
    {TensorFamily::SelfAttnQProj, SourceClass::Language, kLayerPrefix,
     ".self_attn.q_proj.weight", true, false, true,
     shape2(2 * kQueryHeads * kHeadDim, kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::SelfAttnKProj, SourceClass::Language, kLayerPrefix,
     ".self_attn.k_proj.weight", true, false, true,
     shape2(kKvHeads * kHeadDim, kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::SelfAttnVProj, SourceClass::Language, kLayerPrefix,
     ".self_attn.v_proj.weight", true, false, true,
     shape2(kKvHeads * kHeadDim, kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::SelfAttnOProj, SourceClass::Language, kLayerPrefix,
     ".self_attn.o_proj.weight", true, false, true, shape2(kHidden, kZWidth),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::SelfAttnQNorm, SourceClass::Language, kLayerPrefix,
     ".self_attn.q_norm.weight", true, false, true, shape1(kHeadDim),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::QkNorm,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::SelfAttnKNorm, SourceClass::Language, kLayerPrefix,
     ".self_attn.k_norm.weight", true, false, true, shape1(kHeadDim),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::QkNorm,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::MlpGateProj, SourceClass::Language, kLayerPrefix,
     ".mlp.gate_proj.weight", true, false, false,
     shape2(kIntermediate, kHidden), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::Mlp},
    {TensorFamily::MlpUpProj, SourceClass::Language, kLayerPrefix,
     ".mlp.up_proj.weight", true, false, false, shape2(kIntermediate, kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::Mlp},
    {TensorFamily::MlpDownProj, SourceClass::Language, kLayerPrefix,
     ".mlp.down_proj.weight", true, false, false,
     shape2(kHidden, kIntermediate), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::Mlp},
    {TensorFamily::MtpFc, SourceClass::MtpRetainedDisabled, "mtp.fc.weight", "",
     false, false, false, shape2(kHidden, 2 * kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::MtpMix},
    {TensorFamily::MtpNorm, SourceClass::MtpRetainedDisabled, "mtp.norm.weight",
     "", false, false, false, shape1(kHidden),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::MtpNorm,
     SemanticNodeKind::LmHead},
    {TensorFamily::MtpPreFcNormEmbedding, SourceClass::MtpRetainedDisabled,
     "mtp.pre_fc_norm_embedding.weight", "", false, false, false,
     shape1(kHidden), PhysicalLayoutId::CudaBf16VectorV0,
     TensorRole::AdditiveNorm,
     SemanticNodeKind::MtpMix},
    {TensorFamily::MtpPreFcNormHidden, SourceClass::MtpRetainedDisabled,
     "mtp.pre_fc_norm_hidden.weight", "", false, false, false, shape1(kHidden),
     PhysicalLayoutId::CudaBf16VectorV0, TensorRole::AdditiveNorm,
     SemanticNodeKind::MtpMix},
    {TensorFamily::MtpInputLayernorm, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.input_layernorm.weight", "", false, false, false,
     shape1(kHidden), PhysicalLayoutId::CudaBf16VectorV0,
     TensorRole::AdditiveNorm,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::MtpPostAttentionLayernorm, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.post_attention_layernorm.weight", "", false, false, false,
     shape1(kHidden), PhysicalLayoutId::CudaBf16VectorV0,
     TensorRole::AdditiveNorm,
     SemanticNodeKind::Mlp},
    {TensorFamily::MtpSelfAttnQProj, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.self_attn.q_proj.weight", "", false, false, false,
     shape2(2 * kQueryHeads * kHeadDim, kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::MtpSelfAttnKProj, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.self_attn.k_proj.weight", "", false, false, false,
     shape2(kKvHeads * kHeadDim, kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::MtpSelfAttnVProj, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.self_attn.v_proj.weight", "", false, false, false,
     shape2(kKvHeads * kHeadDim, kHidden),
     PhysicalLayoutId::CudaBf16DenseTileV0, TensorRole::DenseWeight,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::MtpSelfAttnOProj, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.self_attn.o_proj.weight", "", false, false, false,
     shape2(kHidden, kZWidth), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::GatedAttention},
    {TensorFamily::MtpSelfAttnQNorm, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.self_attn.q_norm.weight", "", false, false, false,
     shape1(kHeadDim), PhysicalLayoutId::CudaBf16VectorV0, TensorRole::QkNorm,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::MtpSelfAttnKNorm, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.self_attn.k_norm.weight", "", false, false, false,
     shape1(kHeadDim), PhysicalLayoutId::CudaBf16VectorV0, TensorRole::QkNorm,
     SemanticNodeKind::GatedAttention},
    {TensorFamily::MtpMlpGateProj, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.mlp.gate_proj.weight", "", false, false, false,
     shape2(kIntermediate, kHidden), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::Mlp},
    {TensorFamily::MtpMlpUpProj, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.mlp.up_proj.weight", "", false, false, false,
     shape2(kIntermediate, kHidden), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::Mlp},
    {TensorFamily::MtpMlpDownProj, SourceClass::MtpRetainedDisabled,
     "mtp.layers.0.mlp.down_proj.weight", "", false, false, false,
     shape2(kHidden, kIntermediate), PhysicalLayoutId::CudaBf16DenseTileV0,
     TensorRole::DenseWeight, SemanticNodeKind::Mlp},
};

ExpectedTensor make_expected(FamilyIdentity const& fam, std::string name,
                             std::uint32_t layer) {
  ExpectedTensor t{};
  t.name = std::move(name);
  t.family = fam.family;
  t.source_class = fam.source_class;
  t.shape = fam.shape;
  t.layout = fam.layout;
  t.role = fam.role;
  t.node = fam.node;
  t.layer_index = fam.source_class == SourceClass::MtpRetainedDisabled
                      ? 0
                      : layer;
  if (fam.family == TensorFamily::InputLayernorm && layer != kNoLayerIndex &&
      is_full_attention_layer(layer)) {
    t.node = SemanticNodeKind::GatedAttention;
  }
  if (fam.source_class == SourceClass::MtpRetainedDisabled &&
      fam.family == TensorFamily::MtpInputLayernorm) {
    t.node = SemanticNodeKind::GatedAttention;
  }
  return t;
}

std::string layer_name(FamilyIdentity const& fam, std::uint32_t layer) {
  return std::string(fam.prefix) + std::to_string(layer) + fam.suffix;
}

bool layer_applies(FamilyIdentity const& fam, std::uint32_t layer) {
  if (fam.linear_layers_only && is_full_attention_layer(layer)) {
    return false;
  }
  if (fam.full_layers_only && !is_full_attention_layer(layer)) {
    return false;
  }
  return true;
}

bool exact_source_sharing_allowed(SourceTensor const& a,
                                  SourceTensor const& b) noexcept {
  // Vision payloads are outside the emitted artifact, so exact aliases among
  // them are harmless. Every language/MTP source tensor has distinct storage;
  // their artifact-level MTP aliases are generated separately.
  return is_vision_tensor(a.name) && is_vision_tensor(b.name);
}

std::expected<std::uint64_t, CompilerError> source_tensor_bytes(
    SourceTensor const& tensor) {
  if (tensor.shape.empty() || tensor.shape.size() > qw38::format::kMaxRank) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                      tensor.name, "shape rank must be 1..8"));
  }
  std::uint64_t elements = 1;
  for (auto const dim : tensor.shape) {
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
  if (elements > std::numeric_limits<std::uint64_t>::max() / 2) {
    return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                      tensor.name,
                                      "BF16 byte size overflows uint64"));
  }
  return elements * 2;
}

}  // namespace

std::span<FamilyIdentity const> family_identity_table() noexcept {
  return kTable;
}

std::vector<ExpectedTensor> expand_identity_table() {
  std::vector<ExpectedTensor> out;
  out.reserve(kIncludedTensors);
  for (auto const& fam : kTable) {
    if (!fam.layer_indexed) {
      out.push_back(make_expected(fam, fam.prefix, kNoLayerIndex));
      continue;
    }
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
      if (!layer_applies(fam, layer)) {
        continue;
      }
      out.push_back(make_expected(fam, layer_name(fam, layer), layer));
    }
  }
  return out;
}

std::vector<VisionTensorIdentity> expand_vision_inventory() {
  std::vector<VisionTensorIdentity> out;
  out.reserve(kVisionTensors);
  constexpr char const kShard[] = "model-00001-of-00018.safetensors";
  struct BlockTensor {
    char const* suffix;
    TensorShapeSpec shape;
  };
  constexpr BlockTensor kBlockTensors[] = {
      {"attn.proj.bias", shape1(1152)},
      {"attn.proj.weight", shape2(1152, 1152)},
      {"attn.qkv.bias", shape1(3456)},
      {"attn.qkv.weight", shape2(3456, 1152)},
      {"mlp.linear_fc1.bias", shape1(4304)},
      {"mlp.linear_fc1.weight", shape2(4304, 1152)},
      {"mlp.linear_fc2.bias", shape1(1152)},
      {"mlp.linear_fc2.weight", shape2(1152, 4304)},
      {"norm1.bias", shape1(1152)},
      {"norm1.weight", shape1(1152)},
      {"norm2.bias", shape1(1152)},
      {"norm2.weight", shape1(1152)},
  };
  for (std::uint32_t block = 0; block < 27; ++block) {
    for (auto const& tensor : kBlockTensors) {
      out.push_back({.name = "model.visual.blocks." + std::to_string(block) +
                             "." + tensor.suffix,
                     .shape = tensor.shape,
                     .shard = kShard});
    }
  }
  auto add = [&](char const* suffix, TensorShapeSpec shape) {
    out.push_back({.name = std::string("model.visual.") + suffix,
                   .shape = shape,
                   .shard = kShard});
  };
  add("merger.linear_fc1.bias", shape1(4608));
  add("merger.linear_fc1.weight", shape2(4608, 4608));
  add("merger.linear_fc2.bias", shape1(5120));
  add("merger.linear_fc2.weight", shape2(5120, 4608));
  add("merger.norm.bias", shape1(1152));
  add("merger.norm.weight", shape1(1152));
  add("patch_embed.proj.bias", shape1(1152));
  add("patch_embed.proj.weight", TensorShapeSpec{.rank = 5,
                                                     .dims = {1152, 3, 2, 16, 16}});
  add("pos_embed.weight", shape2(2304, 1152));
  return out;
}

bool is_vision_tensor(std::string_view name) {
  static auto const inventory = expand_vision_inventory();
  return std::ranges::any_of(inventory, [name](VisionTensorIdentity const& item) {
    return item.name == name;
  });
}

std::expected<ClassifiedCheckpoint, CompilerError> classify_source_tensors(
    std::vector<SourceTensor> tensors) {
  std::unordered_map<std::string, std::size_t> by_name;
  by_name.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    auto const& t = tensors[i];
    if (t.name.empty()) {
      return std::unexpected(make_error(CompilerErrorCode::InvalidConfig,
                                        "tensor.name", "empty tensor name"));
    }
    auto [it, inserted] = by_name.emplace(t.name, i);
    if (!inserted) {
      return std::unexpected(make_error(CompilerErrorCode::DuplicateTensor,
                                        t.name, "duplicate tensor name"));
    }
    auto bytes = source_tensor_bytes(t);
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
  }

  struct SourceSpan {
    SourceTensor const* tensor{};
    std::uint64_t end{};
  };
  std::map<std::string, std::vector<SourceSpan>> spans_by_shard;
  for (auto const& tensor : tensors) {
    if (tensor.data_offset >
        std::numeric_limits<std::uint64_t>::max() - tensor.nbytes) {
      return std::unexpected(make_error(
          CompilerErrorCode::Unrepresentable, tensor.name,
          "source tensor interval overflows uint64"));
    }
    spans_by_shard[tensor.shard].push_back(
        SourceSpan{.tensor = &tensor,
                   .end = tensor.data_offset + tensor.nbytes});
  }

  for (auto& [shard, spans] : spans_by_shard) {
    std::ranges::sort(spans, {}, [](SourceSpan const& span) {
      return std::pair{span.tensor->data_offset, span.end};
    });
    for (std::size_t i = 0; i < spans.size(); ++i) {
      auto const& current = spans[i];
      for (std::size_t j = i; j-- > 0;) {
        auto const& prior = spans[j];
        if (prior.end <= current.tensor->data_offset) {
          continue;
        }
        bool const exact =
            prior.tensor->data_offset == current.tensor->data_offset &&
            prior.end == current.end;
        if (exact &&
            exact_source_sharing_allowed(*prior.tensor, *current.tensor)) {
          continue;
        }
        std::ostringstream detail;
        detail << (exact ? "unexpected exact source sharing"
                         : "overlapping source tensor intervals")
               << " in " << shard << ": " << prior.tensor->name << " and "
               << current.tensor->name;
        return std::unexpected(make_error(
            CompilerErrorCode::UnexpectedSharing, current.tensor->name,
            detail.str()));
      }
    }
  }

  auto expected = expand_identity_table();
  if (expected.size() != kIncludedTensors) {
    return std::unexpected(make_error(CompilerErrorCode::Internal, "identity",
                                      "identity table expansion is not 866"));
  }

  ClassifiedCheckpoint classified{};
  classified.included.reserve(expected.size());
  std::unordered_map<std::string, bool> seen_expected;
  seen_expected.reserve(expected.size());

  for (auto& exp : expected) {
    seen_expected.emplace(exp.name, false);
    auto it = by_name.find(exp.name);
    if (it == by_name.end()) {
      return std::unexpected(make_error(CompilerErrorCode::MissingTensor,
                                        exp.name, "required language/MTP tensor is absent"));
    }
    auto const& src = tensors[it->second];
    seen_expected[exp.name] = true;
    if (src.dtype != "BF16") {
      return std::unexpected(make_error(CompilerErrorCode::DtypeMismatch,
                                        src.name, "expected BF16"));
    }
    if (!exp.shape.matches(src.shape)) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                        src.name, "shape does not match identity table"));
    }
    auto const bytes = source_tensor_bytes(src);
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
    if (src.nbytes != *bytes) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                        src.name,
                                        "payload byte length is not BF16 numel"));
    }
    classified.included.push_back(
        ClassifiedTensor{.expected = std::move(exp), .source = src});
  }

  auto vision = expand_vision_inventory();
  if (vision.size() != kVisionTensors) {
    return std::unexpected(make_error(CompilerErrorCode::Internal, "vision",
                                      "authoritative vision inventory is not 333"));
  }
  std::unordered_map<std::string, VisionTensorIdentity const*> expected_vision;
  expected_vision.reserve(vision.size());
  for (auto const& item : vision) {
    expected_vision.emplace(item.name, &item);
  }
  for (auto const& item : vision) {
    auto const it = by_name.find(item.name);
    if (it == by_name.end()) {
      return std::unexpected(make_error(CompilerErrorCode::MissingTensor,
                                        item.name,
                                        "required excluded vision tensor is absent"));
    }
    auto const& src = tensors[it->second];
    if (src.dtype != "BF16") {
      return std::unexpected(make_error(CompilerErrorCode::DtypeMismatch, src.name,
                                        "excluded vision tensor must be BF16"));
    }
    if (!item.shape.matches(src.shape)) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, src.name,
                                        "shape does not match vision inventory"));
    }
    if (src.shard != item.shard) {
      return std::unexpected(make_error(CompilerErrorCode::ArchitectureMismatch,
                                        src.name,
                                        "vision tensor shard does not match inventory"));
    }
    auto const bytes = source_tensor_bytes(src);
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
    if (src.nbytes != *bytes) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, src.name,
                                        "vision payload byte length is not BF16 numel"));
    }
    ++classified.vision_excluded;
  }

  for (auto const& t : tensors) {
    if (seen_expected.contains(t.name)) {
      continue;
    }
    if (expected_vision.contains(t.name)) {
      continue;
    }
    return std::unexpected(make_error(CompilerErrorCode::ExtraTensor, t.name,
                                      "tensor is not in the V0 identity table"));
  }

  if (classified.vision_excluded != kVisionTensors ||
      tensors.size() != kCheckpointTensors) {
    return std::unexpected(make_error(
        CompilerErrorCode::ArchitectureMismatch, "vision",
        "checkpoint occupancy does not match authoritative inventory"));
  }

  return classified;
}

}  // namespace qw38::compiler
