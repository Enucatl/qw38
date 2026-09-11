#pragma once

// OPT-046 production Q4_K decode path pin. Host-only; do not include the
// integer-dot device header from quant_mmv.cu (that TU owns packed FP32).

#include <cstring>

namespace qw38::cuda {

constexpr char kLegalQ4DecodePathPacked[] = "packed";
constexpr char kLegalQ4DecodePathIntegerQ8[] = "integer_q8";
constexpr char kLegalQ4DecodePathIntegerQ81[] = "integer_q8_1";

// OPT-072 typed Q8_1 pairing: the UseQ81 Q4 consumer treats stored q8_sum as
// integer sum(q). A sum(x) producer must not feed that consumer. Production
// Q8/Q6 paths do not read the stored sum field; leave their staging unchanged.
constexpr char kQ81ConsumerExpects[] = "quartz_q8_1_sum_q";
constexpr char kQ81SumQProducer[] = "quantize_bf16_q8_1";
constexpr char kQ81SumXProducer[] = "quantize_bf16_q8_1_sum_x";
constexpr char kQ81IllegalPairing[] = "sum_x_producer_to_sum_q_consumer";

// Actual launch variants recorded by the kernels that ran, not selector strings.
constexpr char kQ4LaunchVariantPacked[] = "quant_mmv_packed";
constexpr char kQ4LaunchVariantPackedPrequant[] = "quant_mmv_prequant_packed";
constexpr char kQ4LaunchVariantCoopQ8[] = "q4k_coop_mmv_q8";
constexpr char kQ4LaunchVariantCoopQ8Prequant[] = "q4k_coop_mmv_prequant_q8";
constexpr char kQ4LaunchVariantCoopQ81[] = "q4k_coop_mmv_q8_1";
constexpr char kQ4LaunchVariantCoopQ81Prequant[] = "q4k_coop_mmv_prequant_q8_1";
constexpr char kQ4LaunchVariantPairedStaged[] = "q4k_gate_up_swiglu_prequant";
constexpr char kQ4LaunchVariantPaired[] = "q4k_gate_up_swiglu";
constexpr char kQ4LaunchVariantPairedIntegerQ8[] =
    "q4k_coop_gate_up_swiglu_prequant_q8";
constexpr char kQ4LaunchVariantPairedIntegerTraceUnfused[] =
    "q4k_coop_mmv_prequant_q8_trace_unfused";

constexpr int kQ4LaunchTraceCapacity = 8;

// Production pin. Keep sitting may switch away from packed; reject restores it.
constexpr char kSelectedQ4DecodePath[] = "packed";
constexpr unsigned int kSelectedQ4DecodeWarpsPerRow = 4;

inline thread_local const char* g_q4_decode_path_override = nullptr;
inline thread_local unsigned int g_q4_decode_warps_override = 0;
inline thread_local const char* g_last_q4_launch_variant = "";
inline thread_local const char* g_q4_launch_trace[kQ4LaunchTraceCapacity]{};
inline thread_local int g_q4_launch_trace_count = 0;

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

// Q8Block integer cooperative dots. Distinct from half-scale Q8_1.
inline bool q4_decode_uses_integer_q8block() noexcept {
  return std::strcmp(effective_q4_decode_path(),
                     kLegalQ4DecodePathIntegerQ8) == 0;
}

inline void record_q4_launch_variant(const char* variant) noexcept {
  g_last_q4_launch_variant = variant != nullptr ? variant : "";
  if (g_q4_launch_trace_count >= 0 &&
      g_q4_launch_trace_count < kQ4LaunchTraceCapacity) {
    g_q4_launch_trace[g_q4_launch_trace_count] = g_last_q4_launch_variant;
    ++g_q4_launch_trace_count;
  }
}

inline const char* last_q4_launch_variant() noexcept {
  return g_last_q4_launch_variant != nullptr ? g_last_q4_launch_variant : "";
}

inline void clear_q4_launch_trace() noexcept {
  g_last_q4_launch_variant = "";
  g_q4_launch_trace_count = 0;
  for (int index = 0; index < kQ4LaunchTraceCapacity; ++index) {
    g_q4_launch_trace[index] = "";
  }
}

inline int q4_launch_trace_count() noexcept { return g_q4_launch_trace_count; }

inline const char* q4_launch_trace_at(int index) noexcept {
  if (index < 0 || index >= g_q4_launch_trace_count) return "";
  return g_q4_launch_trace[index] != nullptr ? g_q4_launch_trace[index] : "";
}

inline void set_q4_decode_path_override(const char* path,
                                        unsigned int warps_per_row) noexcept {
  g_q4_decode_path_override = path;
  g_q4_decode_warps_override = warps_per_row;
}

inline bool apply_q4_decode_ident(const char* path,
                                  unsigned int warps_per_row) noexcept {
  if (!legal_q4_decode_path(path) ||
      !legal_q4_decode_warps_per_row(warps_per_row)) {
    return false;
  }
  set_q4_decode_path_override(path, warps_per_row);
  return true;
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
