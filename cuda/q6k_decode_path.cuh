#pragma once

// OPT-048 production Q6_K vocabulary-projection path pin. Host-only; do
// not include the integer-dot device header from quant_mmv.cu (that TU
// owns packed FP32).

#include <cstddef>
#include <cstring>

namespace qw38::cuda {

constexpr char kLegalQ6DecodePathPacked[] = "packed";
constexpr char kLegalQ6DecodePathIntegerQ8[] = "integer_q8";
constexpr char kLegalQ6DecodePathIntegerQ81[] = "integer_q8_1";

// Production pin. Keep sitting may switch away from packed; reject restores it.
constexpr char kSelectedQ6DecodePath[] = "integer_q8_1";
constexpr unsigned int kSelectedQ6DecodeWarpsPerRow = 2;

inline thread_local const char* g_q6_decode_path_override = nullptr;
inline thread_local unsigned int g_q6_decode_warps_override = 0;

inline bool legal_q6_decode_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalQ6DecodePathPacked) == 0 ||
          std::strcmp(path, kLegalQ6DecodePathIntegerQ8) == 0 ||
          std::strcmp(path, kLegalQ6DecodePathIntegerQ81) == 0);
}

inline bool legal_q6_decode_warps_per_row(unsigned int warps) noexcept {
  return warps == 1 || warps == 2 || warps == 4 || warps == 8;
}

inline const char* selected_q6_decode_path() noexcept {
  return kSelectedQ6DecodePath;
}

inline unsigned int selected_q6_decode_warps_per_row() noexcept {
  return kSelectedQ6DecodeWarpsPerRow;
}

inline const char* effective_q6_decode_path() noexcept {
  return g_q6_decode_path_override != nullptr ? g_q6_decode_path_override
                                              : kSelectedQ6DecodePath;
}

inline unsigned int effective_q6_decode_warps_per_row() noexcept {
  return g_q6_decode_warps_override > 0 ? g_q6_decode_warps_override
                                        : kSelectedQ6DecodeWarpsPerRow;
}

inline bool q6_decode_path_is_integer(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalQ6DecodePathIntegerQ8) == 0 ||
          std::strcmp(path, kLegalQ6DecodePathIntegerQ81) == 0);
}

inline bool q6_decode_uses_integer() noexcept {
  return q6_decode_path_is_integer(effective_q6_decode_path());
}

// Production integer dots are the 248320-row vocabulary projection. Small
// Q6_K probes keep packed FP32 so the 3e-4 quant_mmv scalar contract stays.
// A/B overrides still force integer on every row count.
constexpr std::size_t kQ6IntegerMinRows = 248320;

inline bool q6_decode_uses_integer_for_rows(std::size_t rows) noexcept {
  if (!q6_decode_uses_integer()) return false;
  if (g_q6_decode_path_override != nullptr) return true;
  return rows >= kQ6IntegerMinRows;
}

inline bool q6_decode_uses_q8_1() noexcept {
  return std::strcmp(effective_q6_decode_path(),
                     kLegalQ6DecodePathIntegerQ81) == 0;
}

inline void set_q6_decode_path_override(const char* path,
                                        unsigned int warps_per_row) noexcept {
  g_q6_decode_path_override = path;
  g_q6_decode_warps_override = warps_per_row;
}

inline void clear_q6_decode_path_override() noexcept {
  g_q6_decode_path_override = nullptr;
  g_q6_decode_warps_override = 0;
}

struct Q6DecodePathScope final {
  Q6DecodePathScope(const char* path, unsigned int warps_per_row) noexcept {
    set_q6_decode_path_override(path, warps_per_row);
  }
  ~Q6DecodePathScope() { clear_q6_decode_path_override(); }
  Q6DecodePathScope(const Q6DecodePathScope&) = delete;
  Q6DecodePathScope& operator=(const Q6DecodePathScope&) = delete;
};

}  // namespace qw38::cuda
