#pragma once

// OPT-147 Quartz-owned llama-shaped 8x8 prompt-attention tile screen over
// shipping OPT-111 `opt111_base` (16x2). Keeps BF16 authoritative state,
// OPT-111 arithmetic (convert-once + decreasing-granularity cp.async KV
// load, no XOR), causal absolute>qpos masks, chunking and final-token
// output policy. GQA ratio six is padded to ncols2=8; heads 6 and 7 are
// masked. Diagnostic pipeline identity `prefill_attention_8x8_v1` lives in
// fattn_mma_f16_pipeline.cuh. Production pin stays opt111_base until keep.
// Not a vendor of ggml tensor/runtime. Mechanism remains unknown unless
// separately evidenced.

#include "attention_decode.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace qw38::cuda {
namespace opt147 {

constexpr char kControlId[] = "opt111_base";
constexpr char kCandidateId[] = "prefill_attention_8x8_v1";
constexpr char kLaunchControl[] = "fattn_mma_pipeline_opt111_base";
constexpr char kLaunchCandidate[] =
    "fattn_mma_pipeline_prefill_attention_8x8_v1";
constexpr char kLlamaRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr int kDkq = 256;
constexpr int kDv = 256;
constexpr int kNcols = 64;
constexpr int kNcols1 = 8;
constexpr int kNcols2 = 8;
constexpr int kGqaRatio = 6;
constexpr int kNthreads = 128;
constexpr int kOccupancy = 1;
constexpr int kNbatchFa = 32;
constexpr int kNstages = 2;
constexpr int kKvParts = 2;
constexpr bool kConvertKvOnce = true;
constexpr bool kLlamaLoad = true;
constexpr bool kXorSwizzle = false;
constexpr int kCorrectnessRows[] = {1, 32, 128, 2048, 4096};
constexpr int kCorrectnessRowCount = 5;

inline bool is_candidate_path(const char* path) noexcept {
  return path != nullptr && std::strcmp(path, kCandidateId) == 0;
}

inline const char* launch_for(const char* path) noexcept {
  if (is_candidate_path(path)) return kLaunchCandidate;
  if (path != nullptr && std::strcmp(path, kControlId) == 0) return kLaunchControl;
  return "fattn_mma_pipeline_opt111_base";
}

}  // namespace opt147
}  // namespace qw38::cuda
