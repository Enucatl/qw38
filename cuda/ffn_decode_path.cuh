#pragma once

// OPT-049 production decode-FFN gate/up path pin. Host-only.

#include <cstring>

namespace qw38::cuda {

constexpr char kLegalFfnDecodePathSeparate[] = "separate";
constexpr char kLegalFfnDecodePathSharedStage[] = "shared_stage";
constexpr char kLegalFfnDecodePathPaired[] = "paired";
constexpr char kLegalFfnDecodePathPairedStaged[] = "paired_staged";

// Production pin. Keep sitting may switch away from separate; reject restores it.
constexpr char kSelectedFfnDecodePath[] = "paired_staged";

inline thread_local const char* g_ffn_decode_path_override = nullptr;

inline bool legal_ffn_decode_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalFfnDecodePathSeparate) == 0 ||
          std::strcmp(path, kLegalFfnDecodePathSharedStage) == 0 ||
          std::strcmp(path, kLegalFfnDecodePathPaired) == 0 ||
          std::strcmp(path, kLegalFfnDecodePathPairedStaged) == 0);
}

inline const char* selected_ffn_decode_path() noexcept {
  return kSelectedFfnDecodePath;
}

inline const char* effective_ffn_decode_path() noexcept {
  return g_ffn_decode_path_override != nullptr ? g_ffn_decode_path_override
                                               : kSelectedFfnDecodePath;
}

inline bool ffn_decode_shares_stage() noexcept {
  const char* path = effective_ffn_decode_path();
  return std::strcmp(path, kLegalFfnDecodePathSharedStage) == 0 ||
         std::strcmp(path, kLegalFfnDecodePathPairedStaged) == 0;
}

inline bool ffn_decode_uses_paired() noexcept {
  const char* path = effective_ffn_decode_path();
  return std::strcmp(path, kLegalFfnDecodePathPaired) == 0 ||
         std::strcmp(path, kLegalFfnDecodePathPairedStaged) == 0;
}

inline bool ffn_decode_uses_bf16_activations() noexcept {
  return std::strcmp(effective_ffn_decode_path(), kLegalFfnDecodePathPaired) ==
         0;
}

inline void set_ffn_decode_path_override(const char* path) noexcept {
  g_ffn_decode_path_override = path;
}

inline void clear_ffn_decode_path_override() noexcept {
  g_ffn_decode_path_override = nullptr;
}

struct FfnDecodePathScope final {
  explicit FfnDecodePathScope(const char* path) noexcept {
    set_ffn_decode_path_override(path);
  }
  ~FfnDecodePathScope() { clear_ffn_decode_path_override(); }
  FfnDecodePathScope(const FfnDecodePathScope&) = delete;
  FfnDecodePathScope& operator=(const FfnDecodePathScope&) = delete;
};

}  // namespace qw38::cuda
