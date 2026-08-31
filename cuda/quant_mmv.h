#ifndef QW38_CUDA_QUANT_MMV_H_
#define QW38_CUDA_QUANT_MMV_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {

enum class QuantKind : std::uint8_t { kQ4K, kQ6K };

struct Q8Block {
  float scale;
  std::int8_t values[32];
};
static_assert(sizeof(Q8Block) == 36, "unexpected transient Q8 block padding");

std::size_t q8_workspace_bytes(std::size_t columns) noexcept;

cudaError_t launch_quant_mmv(QuantKind kind, const std::uint8_t* weights,
                             std::size_t rows, std::size_t columns,
                             const __nv_bfloat16* activation,
                             Q8Block* q8_workspace, float* output,
                             cudaStream_t stream) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_QUANT_MMV_H_
