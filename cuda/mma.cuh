#pragma once

// Focused sm_120 INT8/FP16 mma.sync helpers for Quartz OPT-016.
// Adapted from llama.cpp ggml/src/ggml-cuda/mma.cuh at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT, The ggml authors).
// Not a vendor of the full 1456-line file. Explicit mma.sync PTX is not
// compiler FMA contraction; Makefile NVCCFLAGS --fmad=false stays unchanged.

#include <cstdint>

namespace qw38::cuda {
namespace mma {

constexpr int kWarpSize = 32;

// Ampere/Turing INT8 fragment layouts (I-major). J is counted in 32-bit
// physical elements. A is m16 x k32 int8 (tile<16,8,int>), B is n8 x k32
// int8 (tile<8,8,int>), C is m16 x n8 int32 (tile<16,8,int>).
inline __device__ int tile16x8_i(int lane, int l) {
  return ((l / 2) * 8) + (lane / 4);
}

inline __device__ int tile16x8_j(int lane, int l) {
  return ((lane % 4) * 2) + (l % 2);
}

inline __device__ int tile8x8_i(int lane, int /*l*/) { return lane / 4; }

inline __device__ int tile8x8_j(int lane, int l) {
  return (l * 4) + (lane % 4);
}

inline __device__ int tile16x8_half2_i(int lane, int l) {
  return ((l % 2) * 8) + (lane / 4);
}

inline __device__ int tile16x8_half2_j(int lane, int l) {
  return ((l / 2) * 4) + (lane % 4);
}

inline __device__ void mma_m16n8k16_s8(int d[4], const int a[2], int b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b));
}

inline __device__ void mma_m16n8k32_s8(
    int d[4], const int a[4], const int b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// FP16 MMA with FP32 accumulation for causal attention (m16n8k16).
inline __device__ void mma_m16n8k16_f16_f32(
    float d[4], const std::uint32_t a[4], const std::uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace mma
}  // namespace qw38::cuda
