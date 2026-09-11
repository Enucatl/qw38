#pragma once

// OPT-061 real-input streaming component replay. Diagnostic only.

#include <array>
#include <cstddef>
#include <cstdint>

#include "ffn_decode_path.cuh"
#include "full_scheduler.h"
#include "gdn_decode_path.cuh"
#include "q4k_decode_path.cuh"
#include "q6k_decode_path.cuh"
#include "q8_decode_path.cuh"

namespace qw38::cuda {

constexpr char kOpt061Task[] = "OPT-061";
constexpr char kOpt061ResultPrefix[] = "QW38_OPT061_COMPONENT_REPLAY_RESULT=";
constexpr char kOpt061EvidenceDir[] =
    "evidence/optimization/opt061-component-replay";
constexpr char kOpt061GgufSha[] =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr char kOpt061LlamaRev[] =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kOpt061CalibrationLayers[] = {0, 3, 31, 32};
constexpr std::size_t kOpt061HeldOutLayers[] = {62, 63};
constexpr std::size_t kOpt061CaptureLayerCount = 6;
constexpr std::size_t kOpt061LayerCount = 64;
constexpr std::size_t kOpt061GateUpDownCallsPerToken = 64;
constexpr std::size_t kOpt061SampledOutputRows = 16;
constexpr std::size_t kOpt061ActivationVectors = 4;
constexpr std::size_t kOpt061PromptSampledRows = 4;
constexpr std::size_t kOpt061PromptRows = 4096;
constexpr std::size_t kOpt061GuardBytes = 64;
constexpr char kOpt061TokenGenerator[] = "(42 + index * 997) % 248320";

constexpr std::size_t kOpt061GdnLayers = 48;
constexpr std::size_t kOpt077GdnCapturePositions = 2;

enum class ReplayFamily { kDecodeFfn, kDecodeMixer, kPromptFfn, kDecodeGdn };
enum class CacheMode { kHot, kRotating };

inline const char* replay_family_name(ReplayFamily family) noexcept {
  switch (family) {
    case ReplayFamily::kDecodeFfn:
      return "decode-ffn";
    case ReplayFamily::kDecodeMixer:
      return "decode-mixer";
    case ReplayFamily::kPromptFfn:
      return "prompt-ffn";
    case ReplayFamily::kDecodeGdn:
      return "decode-gdn";
  }
  return "unknown";
}

inline const char* cache_mode_name(CacheMode mode) noexcept {
  return mode == CacheMode::kHot ? "hot" : "rotating";
}

inline bool cache_mode_is_production_acceptance(CacheMode mode) noexcept {
  return mode == CacheMode::kRotating;
}

struct ReplayCallCounts final {
  int gate_up = 0;
  int down = 0;
  int mixer_projections = 0;
  int paired_gate_up_charged_once = 1;
};

inline ReplayCallCounts production_decode_ffn_calls() noexcept {
  ReplayCallCounts counts;
  counts.gate_up = static_cast<int>(kOpt061GateUpDownCallsPerToken);
  counts.down = static_cast<int>(kOpt061GateUpDownCallsPerToken);
  counts.paired_gate_up_charged_once = 1;
  return counts;
}

struct TensorIdentity final {
  char name[80]{};
  char role[32]{};
  char dtype[16]{};
  char sha256[65]{};
  std::size_t layer = 0;
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::size_t storage_bytes = 0;
};

struct ReplayRound final {
  CacheMode cache_mode = CacheMode::kRotating;
  ReplayFamily family = ReplayFamily::kDecodeFfn;
  int sample_index = 0;
  bool warmup = false;
  float enclosing_ms = 0.0F;
  float kernel_only_ms = 0.0F;
  int gate_up_calls = 0;
  int down_calls = 0;
  int mixer_calls = 0;
  std::size_t rotating_layers = 0;
  std::size_t working_set_bytes = 0;
  bool eviction_outside_interval = false;
  bool observation_is_independent_round = true;
};

}  // namespace qw38::cuda
