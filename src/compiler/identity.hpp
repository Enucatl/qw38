#pragma once

#include "compiler/error.hpp"
#include "format/constants.hpp"
#include "format/schema.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qw38::compiler {

inline constexpr std::uint64_t kHidden = 5120;
inline constexpr std::uint64_t kIntermediate = 17408;
inline constexpr std::uint64_t kVocab = 248320;
inline constexpr std::uint32_t kLayers = 64;
inline constexpr std::uint32_t kFullInterval = 4;
inline constexpr std::uint32_t kLinearLayers = 48;
inline constexpr std::uint32_t kFullLayers = 16;
inline constexpr std::uint64_t kHeadDim = 256;
inline constexpr std::uint64_t kQueryHeads = 24;
inline constexpr std::uint64_t kKvHeads = 4;
inline constexpr std::uint64_t kLinearKeyHeads = 16;
inline constexpr std::uint64_t kLinearValueHeads = 48;
inline constexpr std::uint64_t kLinearHeadDim = 128;
inline constexpr std::uint64_t kQkvWidth = 10240;
inline constexpr std::uint64_t kZWidth = 6144;
inline constexpr std::uint64_t kConvKernel = 4;
inline constexpr std::uint64_t kRotaryDim = 64;
inline constexpr std::uint32_t kRopeFreqs = 32;
inline constexpr double kRopeTheta = 10000000.0;
inline constexpr std::uint32_t kLanguageInstances = 130;
inline constexpr std::uint32_t kMtpInstances = 5;
inline constexpr std::uint32_t kIncludedTensors = 866;
inline constexpr std::uint32_t kCheckpointTensors = 1199;
inline constexpr std::uint32_t kVisionTensors = 333;
inline constexpr std::uint32_t kShards = 18;

inline constexpr char const kArchitecture[] = "Qwen3_5ForConditionalGeneration";
inline constexpr char const kModelType[] = "qwen3_5";
inline constexpr char const kTextModelType[] = "qwen3_5_text";
inline constexpr char const kRopeInvFreqName[] = "rope.inv_freq";
inline constexpr char const kMtpEmbedAliasName[] = "mtp.embed_tokens.weight";
inline constexpr char const kMtpLmHeadAliasName[] = "mtp.lm_head.weight";
inline constexpr char const kEmbedName[] =
    "model.language_model.embed_tokens.weight";
inline constexpr char const kLmHeadName[] = "lm_head.weight";
inline constexpr char const kFinalNormName[] = "model.language_model.norm.weight";
inline constexpr char const kCompilerIdent[] = "qw38-bf16-identity";
inline constexpr char const kProductionCompilerIdent[] = "qw38-v0";
inline constexpr std::uint32_t kCompilerMajor = 0;
inline constexpr std::uint32_t kCompilerMinor = 1;
inline constexpr std::uint32_t kCompilerPatch = 0;

[[nodiscard]] constexpr bool is_full_attention_layer(std::uint32_t layer) noexcept {
  return layer < kLayers && (layer % kFullInterval) == (kFullInterval - 1);
}

enum class TensorFamily : std::uint16_t {
  Embed = 1,
  FinalNorm,
  LmHead,
  InputLayernorm,
  PostAttentionLayernorm,
  LinearAttnALog,
  LinearAttnConv1d,
  LinearAttnDtBias,
  LinearAttnInProjA,
  LinearAttnInProjB,
  LinearAttnInProjQkv,
  LinearAttnInProjZ,
  LinearAttnNorm,
  LinearAttnOutProj,
  SelfAttnQProj,
  SelfAttnKProj,
  SelfAttnVProj,
  SelfAttnOProj,
  SelfAttnQNorm,
  SelfAttnKNorm,
  MlpGateProj,
  MlpUpProj,
  MlpDownProj,
  MtpFc,
  MtpNorm,
  MtpPreFcNormEmbedding,
  MtpPreFcNormHidden,
  MtpInputLayernorm,
  MtpPostAttentionLayernorm,
  MtpSelfAttnQProj,
  MtpSelfAttnKProj,
  MtpSelfAttnVProj,
  MtpSelfAttnOProj,
  MtpSelfAttnQNorm,
  MtpSelfAttnKNorm,
  MtpMlpGateProj,
  MtpMlpUpProj,
  MtpMlpDownProj,
  RopeInvFreq,
};

enum class SourceClass : std::uint8_t {
  Language = 1,
  MtpRetainedDisabled = 2,
  VisionExcluded = 3,
  Generated = 4,
};

struct TensorShapeSpec {
  std::uint8_t rank{};
  std::array<std::uint64_t, 3> dims{};

  [[nodiscard]] bool matches(std::span<std::uint64_t const> got) const noexcept {
    if (got.size() != rank) {
      return false;
    }
    for (std::uint8_t i = 0; i < rank; ++i) {
      if (got[i] != dims[i]) {
        return false;
      }
    }
    return true;
  }
};

// Encoded identity: one row per family, expanded over sitting layer sets in
// code rather than inferred from checkpoint strings.
struct FamilyIdentity {
  TensorFamily family{};
  SourceClass source_class{};
  char const* prefix{};
  char const* suffix{};  // empty => prefix is the full exact name
  bool layer_indexed{};
  bool linear_layers_only{};
  bool full_layers_only{};
  TensorShapeSpec shape{};
  qw38::format::PhysicalLayoutId layout{
      qw38::format::PhysicalLayoutId::CudaBf16VectorV0};
  qw38::format::TensorRole role{qw38::format::TensorRole::DenseWeight};
  qw38::format::SemanticNodeKind node{qw38::format::SemanticNodeKind::Mlp};
};

[[nodiscard]] std::span<FamilyIdentity const> family_identity_table() noexcept;

struct ExpectedTensor {
  std::string name;
  TensorFamily family{};
  SourceClass source_class{};
  TensorShapeSpec shape{};
  qw38::format::PhysicalLayoutId layout{};
  qw38::format::TensorRole role{};
  qw38::format::SemanticNodeKind node{};
  std::uint32_t layer_index{qw38::format::kNoLayerIndex};
};

[[nodiscard]] std::vector<ExpectedTensor> expand_identity_table();

[[nodiscard]] bool is_vision_tensor(std::string_view name) noexcept;

struct SourceTensor {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::string shard;
  std::uint64_t data_offset{};
  std::uint64_t nbytes{};
};

struct ClassifiedTensor {
  ExpectedTensor expected;
  SourceTensor source;
};

struct ClassifiedCheckpoint {
  std::vector<ClassifiedTensor> included;
  std::uint32_t vision_excluded{};
  std::uint32_t shard_count{};
};

[[nodiscard]] std::expected<ClassifiedCheckpoint, CompilerError>
classify_source_tensors(std::vector<SourceTensor> tensors);

}  // namespace qw38::compiler
