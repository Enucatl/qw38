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
const char* selected_mmv_load_path() noexcept;
bool mmv_uses_packed_loads() noexcept;
int mmv_packed_occupancy(QuantKind kind, unsigned int warps) noexcept;
unsigned int selected_mmq_prompt_tile(std::size_t prompt_rows) noexcept;
unsigned int selected_mmq_prompt_tile(QuantKind kind,
                                      std::size_t prompt_rows) noexcept;

cudaError_t launch_quant_mmv(QuantKind kind, const std::uint8_t* weights,
                             std::size_t rows, std::size_t columns,
                             const __nv_bfloat16* activation,
                             Q8Block* q8_workspace, float* output,
                             cudaStream_t stream) noexcept;

cudaError_t launch_quantize_bf16_q8(const __nv_bfloat16* activation,
                                    Q8Block* q8, std::size_t columns,
                                    cudaStream_t stream) noexcept;

cudaError_t launch_quant_mmv_prequant(QuantKind kind,
                                      const std::uint8_t* weights,
                                      std::size_t rows, std::size_t columns,
                                      const Q8Block* q8, float* output,
                                      cudaStream_t stream) noexcept;

cudaError_t launch_quant_mmq(QuantKind kind, const std::uint8_t* weights,
                             std::size_t output_rows, std::size_t columns,
                             const __nv_bfloat16* prompt,
                             std::size_t prompt_rows, Q8Block* q8_workspace,
                             float* output, cudaStream_t stream,
                             float* tmp_fixup = nullptr,
                             std::size_t tmp_fixup_floats = 0) noexcept;

cudaError_t launch_quant_mmv_variant(
    QuantKind kind, const std::uint8_t* weights, std::size_t rows,
    std::size_t columns, const __nv_bfloat16* activation,
    Q8Block* q8_workspace, float* output, unsigned int warps,
    cudaStream_t stream) noexcept;

cudaError_t launch_quant_mmv_path(
    QuantKind kind, const std::uint8_t* weights, std::size_t rows,
    std::size_t columns, const __nv_bfloat16* activation,
    Q8Block* q8_workspace, float* output, const char* load_path,
    cudaStream_t stream, unsigned int warps = 0) noexcept;

cudaError_t launch_quant_mmq_variant(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt,
    std::size_t prompt_rows, Q8Block* q8_workspace, float* output,
    unsigned int prompt_tile, cudaStream_t stream) noexcept;

unsigned int selected_mma_mmq_prompt_tile() noexcept;
std::size_t mma_mmq_shared_bytes(unsigned int prompt_tile) noexcept;
int mma_mmq_occupancy(QuantKind kind, unsigned int prompt_tile) noexcept;

const char* selected_mmq_stream_k_path() noexcept;
bool mmq_uses_stream_k() noexcept;
int mmq_stream_k_nsm() noexcept;
int mmq_stream_k_nblocks(int nsm, int ntiles_dst, const char* path) noexcept;
std::size_t mmq_stream_k_fixup_floats(int nblocks, unsigned int i,
                                     unsigned int j) noexcept;
bool mmq_stream_k_fixup_needed(int ntiles_dst, int nblocks) noexcept;
int mmq_stream_k_occupancy(QuantKind kind) noexcept;
int mmq_stream_k_fixup_occupancy() noexcept;

cudaError_t launch_quant_mmq_mma(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    Q8Block* q8_workspace, float* output, cudaStream_t stream,
    float* tmp_fixup = nullptr, std::size_t tmp_fixup_floats = 0) noexcept;

cudaError_t launch_quant_mmq_mma_tile(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    Q8Block* q8_workspace, float* output, unsigned int prompt_tile,
    cudaStream_t stream, float* tmp_fixup = nullptr,
    std::size_t tmp_fixup_floats = 0) noexcept;

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

unsigned int selected_q8_quality_mmq_prompt_tile() noexcept;
int q8_quality_mmq_occupancy(unsigned int prompt_tile) noexcept;
int q8_quality_mmq_occupancy_i(unsigned int prompt_tile,
                               unsigned int quality_i) noexcept;

bool skinny_mixer_q8_output_rows(std::size_t output_rows) noexcept;
const char* selected_skinny_mixer_path() noexcept;

bool large_mixer_q8_output_rows(std::size_t output_rows) noexcept;
std::size_t q8_aligned_soa_bytes(std::size_t output_rows,
                                 std::size_t columns) noexcept;
const char* selected_large_mixer_q8_path() noexcept;
int q8_d2r_occupancy(unsigned int prompt_tile) noexcept;

cudaError_t launch_repack_q8_0_aligned(const std::uint8_t* weights,
                                       std::size_t output_rows,
                                       std::size_t columns, std::uint8_t* soa,
                                       cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_d2r(const std::uint8_t* soa, std::size_t output_rows,
                              std::size_t columns, const Q8Block* y,
                              std::size_t prompt_rows, float* output,
                              cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_quality_mma_i(const std::uint8_t* weights,
                                        std::size_t output_rows,
                                        std::size_t columns, const Q8Block* y,
                                        std::size_t prompt_rows, float* output,
                                        unsigned int quality_i,
                                        cudaStream_t stream) noexcept;

cudaError_t launch_quantize_mmq_q8_1(QuantKind kind, const __nv_bfloat16* prompt,
                                     std::size_t prompt_rows,
                                     std::size_t columns, Q8Block* workspace,
                                     cudaStream_t stream) noexcept;

const char* selected_ffn_path() noexcept;
bool ffn_shares_gate_up_y() noexcept;
bool ffn_swiglu_writes_q8() noexcept;

cudaError_t launch_quant_mmq_mma_y(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const Q8Block* y, std::size_t prompt_rows,
    float* output, cudaStream_t stream, float* tmp_fixup = nullptr,
    std::size_t tmp_fixup_floats = 0) noexcept;

bool legal_ffn_quality_i(unsigned int quality_i) noexcept;
bool legal_ffn_prompt_tile(unsigned int prompt_tile) noexcept;
unsigned int selected_ffn_gate_quality_i() noexcept;
unsigned int selected_ffn_gate_prompt_tile() noexcept;
unsigned int selected_ffn_up_quality_i() noexcept;
unsigned int selected_ffn_up_prompt_tile() noexcept;
unsigned int selected_ffn_down_quality_i() noexcept;
unsigned int selected_ffn_down_prompt_tile() noexcept;
int mma_mmq_occupancy_ij(QuantKind kind, unsigned int prompt_tile,
                         unsigned int quality_i) noexcept;

cudaError_t launch_quant_mmq_mma_y_ij(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const Q8Block* y, std::size_t prompt_rows,
    float* output, unsigned int quality_i, unsigned int prompt_tile,
    cudaStream_t stream, float* tmp_fixup = nullptr,
    std::size_t tmp_fixup_floats = 0) noexcept;

cudaError_t launch_quant_mmq_mma_y_path(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const Q8Block* y, std::size_t prompt_rows,
    float* output, cudaStream_t stream, const char* path, float* tmp_fixup,
    std::size_t tmp_fixup_floats) noexcept;

cudaError_t launch_swiglu_quantize_mmq_q8_1(const float* gate, const float* up,
                                            std::size_t prompt_rows,
                                            std::size_t columns, Q8Block* y,
                                            cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_quality_mma(const std::uint8_t* weights,
                                      std::size_t output_rows,
                                      std::size_t columns, const Q8Block* y,
                                      std::size_t prompt_rows, float* output,
                                      cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_quality(const std::uint8_t* weights,
                                  std::size_t output_rows, std::size_t columns,
                                  const __nv_bfloat16* prompt,
                                  std::size_t prompt_rows, Q8Block* q8_workspace,
                                  float* output, cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_bf16(const std::uint8_t* weights,
                               std::size_t output_rows, std::size_t columns,
                               const __nv_bfloat16* prompt,
                               std::size_t prompt_rows, Q8Block* q8_workspace,
                               float* output, cudaStream_t stream) noexcept;

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
