#pragma once

// OPT-111 Quartz-owned prompt-attention adapter of pinned llama.cpp
// fattn-mma-f16 (MIT, The ggml authors, revision
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328). Diagnostic pipeline identities
// `opt111_base` and `opt111_xor` live in fattn_mma_f16_pipeline.cuh. OPT-111
// kept `opt111_base`; `opt111_xor` remains diagnostic. Not a vendor of ggml
// tensor/runtime machinery.
// XOR swizzle is technique inspiration from current-llama e4b9af007 and is
// screened only after the base path beats kv_once.

#include "attention_decode.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace qw38::cuda {
namespace opt111 {

constexpr char kControlId[] = "kv_once";
constexpr char kBaseId[] = "opt111_base";
constexpr char kXorId[] = "opt111_xor";
constexpr char kLaunchBase[] = "fattn_mma_pipeline_opt111_base";
constexpr char kLaunchXor[] = "fattn_mma_pipeline_opt111_xor";
constexpr char kLlamaRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr int kDkq = 256;
constexpr int kDv = 256;
constexpr int kNcols = 32;
constexpr int kNcols1 = 16;
constexpr int kNcols2 = 2;
constexpr int kNthreads = 128;
constexpr int kOccupancy = 2;
constexpr int kNbatchFa = 32;
constexpr int kNbatchK2 = 128;
constexpr int kNbatchV2 = 128;
constexpr int kNbatchCombine = 128;
constexpr int kNstages = 2;
constexpr bool kQInRegPinned = true;
constexpr int kPinnedKvPadH2 = 4;
constexpr int kPrimitiveRows[] = {1, 32, 128, 512, 2048, 4096};
constexpr int kPrimitiveRowCount = 6;

inline bool is_candidate_path(const char* path) noexcept {
  return path != nullptr && (std::strcmp(path, kBaseId) == 0 ||
                             std::strcmp(path, kXorId) == 0);
}

inline const char* launch_for(const char* path) noexcept {
  if (path != nullptr && std::strcmp(path, kXorId) == 0) return kLaunchXor;
  if (path != nullptr && std::strcmp(path, kBaseId) == 0) return kLaunchBase;
  return "fattn_mma_pipeline_kv_once";
}

}  // namespace opt111
}  // namespace qw38::cuda
