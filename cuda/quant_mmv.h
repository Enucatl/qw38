#ifndef QW38_CUDA_QUANT_MMV_H_
#define QW38_CUDA_QUANT_MMV_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {

enum class QuantKind : std::uint8_t { kQ4K, kQ6K, kQ8_0 };

struct Q8Block {
  float scale;
  std::int8_t values[32];
};
static_assert(sizeof(Q8Block) == 36, "unexpected transient Q8 block padding");

std::size_t q8_workspace_bytes(std::size_t columns) noexcept;
std::size_t q8_prompt_workspace_bytes(std::size_t prompt_rows,
                                      std::size_t columns) noexcept;

unsigned int selected_mmv_warps(std::size_t rows) noexcept;
unsigned int selected_mmq_prompt_tile(std::size_t prompt_rows) noexcept;
unsigned int selected_mmq_prompt_tile(QuantKind kind,
                                      std::size_t prompt_rows) noexcept;

cudaError_t launch_quant_mmv(QuantKind kind, const std::uint8_t* weights,
                             std::size_t rows, std::size_t columns,
                             const __nv_bfloat16* activation,
                             Q8Block* q8_workspace, float* output,
                             cudaStream_t stream) noexcept;

cudaError_t launch_quant_mmq(QuantKind kind, const std::uint8_t* weights,
                             std::size_t output_rows, std::size_t columns,
                             const __nv_bfloat16* prompt,
                             std::size_t prompt_rows, Q8Block* q8_workspace,
                             float* output, cudaStream_t stream) noexcept;

cudaError_t launch_quant_mmv_variant(
    QuantKind kind, const std::uint8_t* weights, std::size_t rows,
    std::size_t columns, const __nv_bfloat16* activation,
    Q8Block* q8_workspace, float* output, unsigned int warps,
    cudaStream_t stream) noexcept;

cudaError_t launch_quant_mmq_variant(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt,
    std::size_t prompt_rows, Q8Block* q8_workspace, float* output,
    unsigned int prompt_tile, cudaStream_t stream) noexcept;

unsigned int selected_mma_mmq_prompt_tile() noexcept;

cudaError_t launch_quant_mmq_mma(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    float* output, cudaStream_t stream) noexcept;

cudaError_t launch_quant_mmq_mma_tile(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    float* output, unsigned int prompt_tile, cudaStream_t stream) noexcept;

unsigned int selected_q8_mma_mmq_prompt_tile() noexcept;
int q8_mma_mmq_occupancy(unsigned int prompt_tile) noexcept;

cudaError_t launch_q8_mmq_mma(const std::uint8_t* weights,
                              std::size_t output_rows, std::size_t columns,
                              const __nv_bfloat16* prompt,
                              std::size_t prompt_rows, float* output,
                              cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_mma_tile(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const __nv_bfloat16* prompt, std::size_t prompt_rows, float* output,
    unsigned int prompt_tile, cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_bf16(const std::uint8_t* weights,
                               std::size_t output_rows, std::size_t columns,
                               const __nv_bfloat16* prompt,
                               std::size_t prompt_rows, float* output,
                               cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_bf16_variant(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const __nv_bfloat16* prompt, std::size_t prompt_rows, float* output,
    unsigned int prompt_tile, cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_bf16_reference(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const __nv_bfloat16* prompt, std::size_t prompt_rows, float* output,
    cudaStream_t stream) noexcept;

cudaError_t launch_quant_row_decode(QuantKind kind,
                                    const std::uint8_t* weights,
                                    std::size_t rows, std::size_t columns,
                                    std::size_t row, __nv_bfloat16* output,
                                    cudaStream_t stream) noexcept;

cudaError_t launch_quant_rows_decode_widen(QuantKind kind,
                                           const std::uint8_t* weights,
                                           std::size_t rows, std::size_t columns,
                                           const std::size_t* token_ids,
                                           std::size_t token_count,
                                           float* output,
                                           cudaStream_t stream) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_QUANT_MMV_H_
