#pragma once

// OPT-110 source-faithful NVIDIA adapter of pinned llama.cpp Q4_K MMVQ
// (MIT, The ggml authors, revision cc83d7b4824f73cfdda4dfbb47ee39804f71b328).
// Namespaced Quartz adapter: Q4_K x Q8_1 one-vector path and fused SwiGLU.
// Does not vendor ggml tensor/graph/allocator/CLI/model code.
//
// Control stays Quartz Q8Block + integer_q8_late (strict NVCCFLAGS).
// This candidate TU is compiled with pinned-llama CUDA flags
// (-O3 --use_fast_math --extended-lambda). The two encodings are not
// interchangeable: never pass Q8Block to MMVQ or block_q8_1 to late.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {
namespace opt110 {

constexpr char kCandidateId[] = "llama_q4k_mmvq";
constexpr char kControlId[] = "integer_q8_late";
constexpr char kLaunchMmvq[] = "opt110_mul_mat_vec_q4_k_q8_1";
constexpr char kLaunchMmvqGlu[] = "opt110_mul_mat_vec_q4_k_q8_1_swiglu";
constexpr char kLaunchQuant[] = "opt110_quantize_bf16_block_q8_1";
constexpr char kLlamaRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr char kLicense[] = "MIT";

constexpr int kWarp = 32;
constexpr int kNwarps = 4;  // GENERIC table, ncols_dst=1
constexpr int kRowsPerBlock = 1;
constexpr int kVdr = 2;     // VDR_Q4_K_Q8_1_MMVQ
constexpr int kQi = 32;     // QI4_K
constexpr int kQk = 256;    // QK_K
constexpr int kQ8 = 32;     // QK8_1
constexpr int kQr4K = 2;
constexpr int kScaleBytes = 12;
constexpr int kQ4KBytes = 144;
constexpr int kQuantizeBlock = 256;
constexpr int kDownRows = 5120;
constexpr int kDownCols = 17408;
constexpr int kGateRows = 17408;
constexpr int kGateCols = 5120;

// Pinned llama block_q8_1: half2 ds=(d, sum(x)), int8 qs[32]. 36 bytes.
// Distinct from Quartz Q8Block {float scale; int8 values[32]}.
struct alignas(4) BlockQ81 {
  __half2 ds;
  std::int8_t qs[32];
};
static_assert(sizeof(BlockQ81) == 36, "block_q8_1 must match llama 36-byte layout");
static_assert(alignof(BlockQ81) <= 4, "block_q8_1 must not gain extra padding");

// Pinned llama block_q4_K: half2 dm, scales[12], qs[128]. 144 bytes.
// Byte-identical to raw GGUF; no replacement device copy.
struct alignas(4) BlockQ4K {
  __half2 dm;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(BlockQ4K) == 144, "block_q4_K must match raw GGUF");

template <int Rows, int Cols>
struct ProductionShape;

template <>
struct ProductionShape<kDownRows, kDownCols> {
  static constexpr int kRows = kDownRows;
  static constexpr int kCols = kDownCols;
  static constexpr const char* kRole = "down";
};

template <>
struct ProductionShape<kGateRows, kGateCols> {
  static constexpr int kRows = kGateRows;
  static constexpr int kCols = kGateCols;
  static constexpr const char* kRole = "gate_up";
};

struct LaunchInfo final {
  int nwarps = kNwarps;
  int rows_per_block = kRowsPerBlock;
  int vdr = kVdr;
  int qi = kQi;
  int threads = kWarp * kNwarps;
  int occupancy = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  std::size_t shared_bytes = 0;
  std::size_t staging_bytes = 0;
  int grid_x = 0;
  int block_x = kWarp;
  int block_y = kNwarps;
  bool fused_glu = false;
  bool used_q8_1 = true;
  bool used_q8block = false;
  const char* launch_name = kLaunchMmvq;
};

const LaunchInfo& last_launch() noexcept;

inline std::size_t q8_1_bytes(std::size_t columns) noexcept {
  return (columns / static_cast<std::size_t>(kQ8)) * sizeof(BlockQ81);
}

cudaError_t launch_quantize_bf16_block_q8_1(const __nv_bfloat16* activation,
                                            BlockQ81* q8, std::size_t columns,
                                            cudaStream_t stream) noexcept;

cudaError_t launch_mmvq_q4k_q8_1(const BlockQ4K* weights, std::size_t rows,
                                 std::size_t columns, const BlockQ81* q8,
                                 float* output, cudaStream_t stream) noexcept;

cudaError_t launch_mmvq_q4k_q8_1_swiglu(const BlockQ4K* gate,
                                        const BlockQ4K* up, std::size_t rows,
                                        std::size_t columns, const BlockQ81* q8,
                                        __nv_bfloat16* output,
                                        cudaStream_t stream) noexcept;

template <int Rows, int Cols>
cudaError_t launch_mmvq_production(const BlockQ4K* weights, const BlockQ81* q8,
                                   float* output, cudaStream_t stream) noexcept {
  using Shape = ProductionShape<Rows, Cols>;
  return launch_mmvq_q4k_q8_1(weights, static_cast<std::size_t>(Shape::kRows),
                              static_cast<std::size_t>(Shape::kCols), q8,
                              output, stream);
}

template <int Rows, int Cols>
cudaError_t launch_mmvq_production_swiglu(const BlockQ4K* gate,
                                          const BlockQ4K* up,
                                          const BlockQ81* q8,
                                          __nv_bfloat16* output,
                                          cudaStream_t stream) noexcept {
  using Shape = ProductionShape<Rows, Cols>;
  return launch_mmvq_q4k_q8_1_swiglu(
      gate, up, static_cast<std::size_t>(Shape::kRows),
      static_cast<std::size_t>(Shape::kCols), q8, output, stream);
}

void mmvq_kernel_attributes(int* registers, std::size_t* local_bytes,
                            int* occupancy) noexcept;
void mmvq_swiglu_kernel_attributes(int* registers, std::size_t* local_bytes,
                                   int* occupancy) noexcept;
void quantize_kernel_attributes(int* registers, std::size_t* local_bytes,
                                int* occupancy) noexcept;

}  // namespace opt110
}  // namespace qw38::cuda
