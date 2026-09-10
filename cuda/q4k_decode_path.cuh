#pragma once

// OPT-046 production Q4_K decode path pin. Host-only; do not include the
// integer-dot device header from quant_mmv.cu (that TU owns packed FP32).

#include <cstring>

namespace qw38::cuda {

constexpr char kLegalQ4DecodePathPacked[] = "packed";
constexpr char kLegalQ4DecodePathIntegerQ8[] = "integer_q8";
constexpr char kLegalQ4DecodePathIntegerQ81[] = "integer_q8_1";

// Production pin. Keep sitting may switch away from packed; reject restores it.
constexpr char kSelectedQ4DecodePath[] = "packed";
constexpr unsigned int kSelectedQ4DecodeWarpsPerRow = 4;

inline thread_local const char* g_q4_decode_path_override = nullptr;
inline thread_local unsigned int g_q4_decode_warps_override = 0;

inline bool legal_q4_decode_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalQ4DecodePathPacked) == 0 ||
          std::strcmp(path, kLegalQ4DecodePathIntegerQ8) == 0 ||
          std::strcmp(path, kLegalQ4DecodePathIntegerQ81) == 0);
}

inline bool legal_q4_decode_warps_per_row(unsigned int warps) noexcept {
  return warps == 1 || warps == 2 || warps == 4 || warps == 8;
}

inline const char* selected_q4_decode_path() noexcept {
  return kSelectedQ4DecodePath;
}

inline unsigned int selected_q4_decode_warps_per_row() noexcept {
  return kSelectedQ4DecodeWarpsPerRow;
}

inline const char* effective_q4_decode_path() noexcept {
  return g_q4_decode_path_override != nullptr ? g_q4_decode_path_override
                                              : kSelectedQ4DecodePath;
}

inline unsigned int effective_q4_decode_warps_per_row() noexcept {
  return g_q4_decode_warps_override > 0 ? g_q4_decode_warps_override
                                        : kSelectedQ4DecodeWarpsPerRow;
}

inline bool q4_decode_path_is_integer(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalQ4DecodePathIntegerQ8) == 0 ||
          std::strcmp(path, kLegalQ4DecodePathIntegerQ81) == 0);
}

inline bool q4_decode_uses_integer() noexcept {
  return q4_decode_path_is_integer(effective_q4_decode_path());
}

inline bool q4_decode_uses_q8_1() noexcept {
  return std::strcmp(effective_q4_decode_path(),
                     kLegalQ4DecodePathIntegerQ81) == 0;
}

inline void set_q4_decode_path_override(const char* path,
                                        unsigned int warps_per_row) noexcept {
  g_q4_decode_path_override = path;
  g_q4_decode_warps_override = warps_per_row;
}

inline void clear_q4_decode_path_override() noexcept {
  g_q4_decode_path_override = nullptr;
  g_q4_decode_warps_override = 0;
}

struct Q4DecodePathScope final {
  Q4DecodePathScope(const char* path, unsigned int warps_per_row) noexcept {
    set_q4_decode_path_override(path, warps_per_row);
  }
  ~Q4DecodePathScope() { clear_q4_decode_path_override(); }
  Q4DecodePathScope(const Q4DecodePathScope&) = delete;
  Q4DecodePathScope& operator=(const Q4DecodePathScope&) = delete;
};

}  // namespace qw38::cuda
