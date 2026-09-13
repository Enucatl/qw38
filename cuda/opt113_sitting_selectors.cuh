#pragma once

// OPT-113 sitting selector parse/apply. Host-includable. Production pins stay
// in their owning headers; this only applies diagnostic overrides for the
// post106_control versus post113_selected sitting.

#include "attention_decode.h"
#include "attention_decode_path.cuh"
#include "gdn_decode_path.cuh"
#include "q4k_decode_path.cuh"

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace qw38::cuda {

constexpr char kOpt113ControlQ4[] = "integer_q8_late";
constexpr char kOpt113SelectedQ4[] = "llama_q4k_mmvq";
constexpr char kOpt113ControlAttention[] = "kv_once";
constexpr char kOpt113SelectedAttention[] = "opt111_base";
constexpr char kOpt113RejectedVec128[] = "vec128_online";
constexpr char kOpt113RejectedGdn[] = "persistent_transposed";
constexpr int kOpt113ControlCrossover = 0;
constexpr int kOpt113SelectedCrossover = 1024;

struct SittingSelectorFlags final {
  const char* q4_decode = nullptr;
  const char* attention_pipeline = nullptr;
  int crossover_threshold = -1;
  bool have_q4 = false;
  bool have_attention = false;
  bool have_crossover = false;
};

inline int sitting_selector_usage(const char* argv0, bool want_prefix) {
  if (want_prefix) {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf PREFIX "
                 "[--q4-decode integer_q8_late|llama_q4k_mmvq] "
                 "[--attention-pipeline kv_once|opt111_base] "
                 "[--decode-attention-crossover-threshold 0|1024]\n",
                 argv0);
  } else {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf "
                 "[--q4-decode integer_q8_late|llama_q4k_mmvq] "
                 "[--attention-pipeline kv_once|opt111_base] "
                 "[--decode-attention-crossover-threshold 0|1024]\n",
                 argv0);
  }
  return 2;
}

inline int parse_sitting_selector_args(int argc, char** argv, const char** model,
                                       std::size_t* prefix,
                                       SittingSelectorFlags* flags) {
  if (argc < 2) return sitting_selector_usage(argv[0], prefix != nullptr);
  *model = nullptr;
  int index = 1;
  if (argv[index][0] != '-') {
    *model = argv[index++];
  }
  if (prefix != nullptr) {
    if (index >= argc || argv[index][0] == '-') {
      return sitting_selector_usage(argv[0], true);
    }
    *prefix = static_cast<std::size_t>(std::atoi(argv[index++]));
  }
  for (; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--q4-decode") == 0 && index + 1 < argc) {
      flags->q4_decode = argv[++index];
      flags->have_q4 = true;
    } else if (std::strcmp(arg, "--attention-pipeline") == 0 &&
               index + 1 < argc) {
      flags->attention_pipeline = argv[++index];
      flags->have_attention = true;
    } else if (std::strcmp(arg, "--decode-attention-crossover-threshold") ==
                   0 &&
               index + 1 < argc) {
      flags->crossover_threshold = std::atoi(argv[++index]);
      flags->have_crossover = true;
    } else {
      return sitting_selector_usage(argv[0], prefix != nullptr);
    }
  }
  if (*model == nullptr) return sitting_selector_usage(argv[0], prefix != nullptr);
  return 0;
}

inline bool apply_sitting_selector_flags(const SittingSelectorFlags& flags) {
  if (flags.have_q4 &&
      !apply_q4_decode_ident(flags.q4_decode, 4U)) {
    std::fprintf(stderr, "invalid --q4-decode %s\n", flags.q4_decode);
    return false;
  }
  if (flags.have_attention &&
      !apply_attention_pipeline_ident(flags.attention_pipeline)) {
    std::fprintf(stderr, "invalid --attention-pipeline %s\n",
                 flags.attention_pipeline);
    return false;
  }
  if (flags.have_crossover &&
      !apply_decode_attention_crossover_threshold(flags.crossover_threshold)) {
    std::fprintf(stderr, "invalid --decode-attention-crossover-threshold %d\n",
                 flags.crossover_threshold);
    return false;
  }
  return true;
}

inline bool opt113_selected_pins_ok() noexcept {
  return std::strcmp(selected_q4_decode_path(), kOpt113SelectedQ4) == 0 &&
         std::strcmp(selected_attention_pipeline_path(),
                     kOpt113SelectedAttention) == 0 &&
         selected_decode_attention_crossover_threshold() ==
             kOpt113SelectedCrossover &&
         std::strcmp(selected_decode_attention_vec128_path(), "warp_query") ==
             0 &&
         std::strcmp(selected_gdn_decode_path(), "sequential") == 0;
}

inline bool opt113_rejected_not_leaked() noexcept {
  return std::strcmp(selected_decode_attention_vec128_path(),
                     kOpt113RejectedVec128) != 0 &&
         std::strcmp(selected_gdn_decode_path(), kOpt113RejectedGdn) != 0;
}

}  // namespace qw38::cuda
