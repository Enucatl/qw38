#ifndef QW38_CUDA_QUANT_MMV_H_
#define QW38_CUDA_QUANT_MMV_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ffn_decode_path.cuh"
#include "production_numerics.h"

namespace qw38::cuda {

enum class QuantKind : std::uint8_t { kQ4K, kQ6K, kQ8_0 };

struct Q8Block {
  float scale;
  std::int8_t values[32];
};
static_assert(sizeof(Q8Block) == 36, "unexpected transient Q8 block padding");

std::size_t q8_workspace_bytes(std::size_t columns) noexcept;
std::size_t q8_1_workspace_bytes(std::size_t columns) noexcept;
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
cudaError_t launch_quantize_bf16_q8_1(const __nv_bfloat16* activation,
                                      void* q8, std::size_t columns,
                                      cudaStream_t stream) noexcept;
cudaError_t launch_quantize_bf16_q8_1_sum_x(const __nv_bfloat16* activation,
                                            void* q8, std::size_t columns,
                                            cudaStream_t stream) noexcept;

int q4k_coop_occupancy(unsigned int warps_per_row, bool q8_1) noexcept;
void q4k_coop_kernel_attributes(unsigned int warps_per_row, bool q8_1,
                                int* registers, std::size_t* local_bytes,
                                int* occupancy) noexcept;

cudaError_t launch_q4k_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int warps_per_row, bool q8_1,
    cudaStream_t stream) noexcept;

// Typed Q8Block prequant. Must not reinterpret the pointer as Q8_1 if a
// global selector changed. Half-scale Q8_1 uses launch_q4k_coop_mmv_prequant
// with a matching Q8_1 buffer, never this entry point.
cudaError_t launch_q4k_coop_mmv_prequant_q8(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const Q8Block* q8, float* output, unsigned int warps_per_row,
    cudaStream_t stream) noexcept;

cudaError_t launch_q4k_coop_gate_up_swiglu_prequant_q8(
    const std::uint8_t* gate_weights, const std::uint8_t* up_weights,
    std::size_t rows, std::size_t columns, const Q8Block* q8,
    __nv_bfloat16* output, unsigned int warps_per_row,
    cudaStream_t stream) noexcept;

int q4k_coop_gate_up_swiglu_occupancy(unsigned int warps_per_row) noexcept;
void q4k_coop_gate_up_swiglu_kernel_attributes(
    unsigned int warps_per_row, int* registers, std::size_t* local_bytes,
    std::size_t* shared_bytes, int* occupancy) noexcept;

cudaError_t launch_q4k_coop_mmv(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const __nv_bfloat16* activation, void* workspace, float* output,
    unsigned int warps_per_row, bool q8_1, cudaStream_t stream) noexcept;

int q6k_coop_occupancy(unsigned int warps_per_row, bool q8_1) noexcept;
void q6k_coop_kernel_attributes(unsigned int warps_per_row, bool q8_1,
                                int* registers, std::size_t* local_bytes,
                                int* occupancy) noexcept;

cudaError_t launch_q6k_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int warps_per_row, bool q8_1,
    cudaStream_t stream) noexcept;

cudaError_t launch_q6k_coop_mmv(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const __nv_bfloat16* activation, void* workspace, float* output,
    unsigned int warps_per_row, bool q8_1, cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmv_bf16(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns,
                               const __nv_bfloat16* activation, float* output,
                               cudaStream_t stream) noexcept;

cudaError_t launch_q8_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int warps_per_row,
    cudaStream_t stream) noexcept;

cudaError_t launch_q8_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int rows_per_cta,
    unsigned int warps_per_row, cudaStream_t stream) noexcept;

cudaError_t launch_q8_coop_mmv(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns,
                               const __nv_bfloat16* activation, void* workspace,
                               float* output, unsigned int warps_per_row,
                               cudaStream_t stream) noexcept;

cudaError_t launch_q8_coop_mmv(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns,
                               const __nv_bfloat16* activation, void* workspace,
                               float* output, unsigned int rows_per_cta,
                               unsigned int warps_per_row,
                               cudaStream_t stream) noexcept;

int q8_coop_occupancy(unsigned int warps_per_row) noexcept;
int q8_coop_occupancy(unsigned int rows_per_cta,
                      unsigned int warps_per_row) noexcept;
void q8_coop_kernel_attributes(unsigned int warps_per_row, int* registers,
                               std::size_t* local_bytes,
                               int* occupancy) noexcept;
void q8_coop_kernel_attributes(unsigned int rows_per_cta,
                               unsigned int warps_per_row, int* registers,
                               std::size_t* local_bytes,
                               int* occupancy) noexcept;
int q8_mmv_bf16_occupancy() noexcept;
void q8_mmv_bf16_kernel_attributes(int* registers, std::size_t* local_bytes,
                                   int* occupancy) noexcept;

cudaError_t launch_quant_mmv_prequant(QuantKind kind,
                                      const std::uint8_t* weights,
                                      std::size_t rows, std::size_t columns,
                                      const Q8Block* q8, float* output,
                                      cudaStream_t stream) noexcept;

int q4k_gate_up_swiglu_occupancy(unsigned int warps, bool staged) noexcept;
void q4k_gate_up_swiglu_kernel_attributes(unsigned int warps, bool staged,
                                          int* registers,
                                          std::size_t* local_bytes,
                                          int* occupancy) noexcept;

cudaError_t launch_q4k_gate_up_swiglu_prequant(
    const std::uint8_t* gate_weights, const std::uint8_t* up_weights,
    std::size_t rows, std::size_t columns, const Q8Block* staged,
    __nv_bfloat16* output, cudaStream_t stream) noexcept;

cudaError_t launch_q4k_gate_up_swiglu(
    const std::uint8_t* gate_weights, const std::uint8_t* up_weights,
    std::size_t rows, std::size_t columns, const __nv_bfloat16* activation,
    Q8Block* workspace, __nv_bfloat16* output, cudaStream_t stream) noexcept;

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

bool legal_mmq_pipeline_path(const char* path) noexcept;
const char* selected_mmq_pipeline_path() noexcept;
const char* effective_mmq_pipeline_path() noexcept;
void set_mmq_pipeline_path_override(const char* path) noexcept;
bool mmq_pipeline_path_on(const char* path) noexcept;
std::size_t mmq_pipeline_extra_shared_bytes(unsigned int prompt_tile,
                                            unsigned int quality_i,
                                            const char* path) noexcept;
int mmq_pipeline_occupancy(QuantKind kind, unsigned int prompt_tile,
                           unsigned int quality_i, const char* path) noexcept;

cudaError_t launch_quant_mmq_mma_y_pipeline(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const Q8Block* y, std::size_t prompt_rows,
    float* output, unsigned int quality_i, unsigned int prompt_tile,
    const char* path, cudaStream_t stream) noexcept;

cudaError_t launch_q8_mmq_quality_mma_pipeline(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const Q8Block* y, std::size_t prompt_rows, float* output,
    unsigned int quality_i, const char* path, cudaStream_t stream) noexcept;

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

bool mmq_q4_pipeline_tile(unsigned int quality_i,
                          unsigned int prompt_tile) noexcept;
const char* mmq_q4_tile_ident(unsigned int quality_i,
                              unsigned int prompt_tile) noexcept;

struct MmqTileDispatch final {
  unsigned int quality_i = 0;
  unsigned int prompt_tile = 0;
  bool aligned = false;
  bool pipeline = false;
  bool fallback = false;
  bool fma = false;
  bool async_y = false;
  const char* ident = "";
  const char* path = "";
  const char* kernel = "";
};

const MmqTileDispatch& last_mmq_tile_dispatch() noexcept;
void clear_mmq_tile_dispatch() noexcept;

cudaError_t mmq_pipeline_kernel_attributes(
    QuantKind kind, unsigned int prompt_tile, unsigned int quality_i,
    const char* path, int* occupancy, int* registers,
    std::size_t* local_bytes, std::size_t* shared_bytes) noexcept;

void set_ffn_tile_override(unsigned int gate_i, unsigned int gate_j,
                           unsigned int up_i, unsigned int up_j,
                           unsigned int down_i, unsigned int down_j) noexcept;
void clear_ffn_tile_override() noexcept;

struct FfnTileOverrideScope final {
  FfnTileOverrideScope(unsigned int gate_i, unsigned int gate_j,
                       unsigned int up_i, unsigned int up_j,
                       unsigned int down_i, unsigned int down_j) noexcept {
    set_ffn_tile_override(gate_i, gate_j, up_i, up_j, down_i, down_j);
  }
  ~FfnTileOverrideScope() { clear_ffn_tile_override(); }
  FfnTileOverrideScope(const FfnTileOverrideScope&) = delete;
  FfnTileOverrideScope& operator=(const FfnTileOverrideScope&) = delete;
};

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
