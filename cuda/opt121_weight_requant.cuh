#pragma once

// OPT-121 selective weight requantization: at most three frozen role/format
// maps from the authenticated Q4_K_M GGUF. Q8/Q6 roles become packed Q4_K
// and reuse existing MMV/MMQ consumers. Native block-scaled FP4 is a bounded
// study and must not relabel Q4_K bits as FP4.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "quant_mmv.h"

namespace qw38::cuda {

enum class WeightRequantConfig : std::uint8_t {
  kNone = 0,
  kQ8ToQ4K = 1,
  kQ8Q6ToQ4K = 2,
  kFp4Study = 3,
};

enum class WeightRole : std::uint8_t {
  kEmbedding = 0,
  kOutput = 1,
  kGdnPackedQkv = 2,
  kGdnValueGate = 3,
  kGdnAlpha = 4,
  kGdnBeta = 5,
  kGdnOutput = 6,
  kAttentionQueryGate = 7,
  kAttentionKey = 8,
  kAttentionValue = 9,
  kAttentionOutput = 10,
  kFfn = 11,
  kOther = 12,
};

constexpr char kLegalWeightRequantNone[] = "none";
constexpr char kLegalWeightRequantQ8ToQ4K[] = "q8_to_q4k";
constexpr char kLegalWeightRequantQ8Q6ToQ4K[] = "q8_q6_to_q4k";
constexpr char kLegalWeightRequantFp4Study[] = "fp4_study";
constexpr char kSelectedWeightRequantConfig[] = "none";
constexpr char kOpt121GgufSha[] =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr char kOpt121PolicyId[] = "opt121_role_map_v1";
constexpr char kOpt121CalibrationId[] = "opt121_calib_v1_seed_20260913";
constexpr char kOpt121Encoding[] = "q4_k_minmax_ref";

inline thread_local int g_weight_requant_override = -1;

inline bool legal_weight_requant_config(WeightRequantConfig config) noexcept {
  return config == WeightRequantConfig::kNone ||
         config == WeightRequantConfig::kQ8ToQ4K ||
         config == WeightRequantConfig::kQ8Q6ToQ4K ||
         config == WeightRequantConfig::kFp4Study;
}

inline WeightRequantConfig weight_requant_from_ident(const char* ident) noexcept {
  if (ident == nullptr) return WeightRequantConfig::kNone;
  if (std::strcmp(ident, kLegalWeightRequantQ8ToQ4K) == 0) {
    return WeightRequantConfig::kQ8ToQ4K;
  }
  if (std::strcmp(ident, kLegalWeightRequantQ8Q6ToQ4K) == 0) {
    return WeightRequantConfig::kQ8Q6ToQ4K;
  }
  if (std::strcmp(ident, kLegalWeightRequantFp4Study) == 0) {
    return WeightRequantConfig::kFp4Study;
  }
  return WeightRequantConfig::kNone;
}

inline const char* weight_requant_ident(WeightRequantConfig config) noexcept {
  switch (config) {
    case WeightRequantConfig::kQ8ToQ4K:
      return kLegalWeightRequantQ8ToQ4K;
    case WeightRequantConfig::kQ8Q6ToQ4K:
      return kLegalWeightRequantQ8Q6ToQ4K;
    case WeightRequantConfig::kFp4Study:
      return kLegalWeightRequantFp4Study;
    case WeightRequantConfig::kNone:
    default:
      return kLegalWeightRequantNone;
  }
}

inline bool legal_weight_requant_ident(const char* ident) noexcept {
  return ident != nullptr &&
         (std::strcmp(ident, kLegalWeightRequantNone) == 0 ||
          std::strcmp(ident, kLegalWeightRequantQ8ToQ4K) == 0 ||
          std::strcmp(ident, kLegalWeightRequantQ8Q6ToQ4K) == 0 ||
          std::strcmp(ident, kLegalWeightRequantFp4Study) == 0);
}

inline WeightRequantConfig selected_weight_requant_config() noexcept {
  return weight_requant_from_ident(kSelectedWeightRequantConfig);
}

inline WeightRequantConfig effective_weight_requant_config() noexcept {
  if (g_weight_requant_override >= 0 &&
      legal_weight_requant_config(
          static_cast<WeightRequantConfig>(g_weight_requant_override))) {
    return static_cast<WeightRequantConfig>(g_weight_requant_override);
  }
  return selected_weight_requant_config();
}

inline void set_weight_requant_override(WeightRequantConfig config) noexcept {
  g_weight_requant_override = static_cast<int>(config);
}

inline void clear_weight_requant_override() noexcept {
  g_weight_requant_override = -1;
}

inline bool apply_weight_requant_ident(const char* ident) noexcept {
  if (!legal_weight_requant_ident(ident)) return false;
  set_weight_requant_override(weight_requant_from_ident(ident));
  return true;
}

struct WeightRequantScope final {
  explicit WeightRequantScope(WeightRequantConfig config) noexcept {
    set_weight_requant_override(config);
  }
  ~WeightRequantScope() { clear_weight_requant_override(); }
  WeightRequantScope(const WeightRequantScope&) = delete;
  WeightRequantScope& operator=(const WeightRequantScope&) = delete;
};

inline bool role_converts_to_q4k(WeightRole role,
                                 WeightRequantConfig config) noexcept {
  if (config != WeightRequantConfig::kQ8ToQ4K &&
      config != WeightRequantConfig::kQ8Q6ToQ4K) {
    return false;
  }
  switch (role) {
    case WeightRole::kGdnPackedQkv:
    case WeightRole::kGdnValueGate:
    case WeightRole::kGdnOutput:
    case WeightRole::kAttentionQueryGate:
    case WeightRole::kAttentionKey:
    case WeightRole::kAttentionValue:
      return true;
    case WeightRole::kAttentionOutput:
    case WeightRole::kOutput:
      return config == WeightRequantConfig::kQ8Q6ToQ4K;
    case WeightRole::kGdnAlpha:
    case WeightRole::kGdnBeta:
    case WeightRole::kEmbedding:
    case WeightRole::kFfn:
    case WeightRole::kOther:
    default:
      return false;
  }
}

inline const char* weight_role_name(WeightRole role) noexcept {
  switch (role) {
    case WeightRole::kEmbedding:
      return "token_embedding";
    case WeightRole::kOutput:
      return "output_projection";
    case WeightRole::kGdnPackedQkv:
      return "gdn_packed_qkv";
    case WeightRole::kGdnValueGate:
      return "gdn_value_gate";
    case WeightRole::kGdnAlpha:
      return "gdn_alpha";
    case WeightRole::kGdnBeta:
      return "gdn_beta";
    case WeightRole::kGdnOutput:
      return "gdn_output";
    case WeightRole::kAttentionQueryGate:
      return "attention_q_gate";
    case WeightRole::kAttentionKey:
      return "attention_k";
    case WeightRole::kAttentionValue:
      return "attention_v";
    case WeightRole::kAttentionOutput:
      return "attention_output";
    case WeightRole::kFfn:
      return "ffn";
    case WeightRole::kOther:
    default:
      return "other";
  }
}

inline std::size_t q4k_storage_bytes(std::size_t rows,
                                     std::size_t columns) noexcept {
  if (rows == 0 || columns == 0 || columns % 256 != 0) return 0;
  return rows * (columns / 256) * 144;
}

}  // namespace qw38::cuda
