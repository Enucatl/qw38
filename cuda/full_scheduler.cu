#include "full_scheduler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>

#include "attention_decode.h"
#include "gdn_step.h"
#include "mixer.h"
#include "pdl_launch.cuh"
#include "rms_norm.cuh"
#include "scheduler.h"
#include "scheduler_primitives.h"
#include "sha256.h"
#include "q8_decode_path.cuh"
#include "ffn_decode_path.cuh"

QW38_PDL_REGISTER_DEVICE_OPS()

namespace qw38::cuda {
namespace {

Status cuda_status(cudaError_t error, const char* message) noexcept;

constexpr std::size_t kGdnLayers = 48;
constexpr std::size_t kAttentionLayers = 16;
constexpr std::size_t kMaximumProjection = internal::kFfnWidth;
#ifdef QW38_DIAGNOSTIC_TRACE
constexpr std::size_t kTraceTapCount = 4;
#endif
constexpr GdnConfig kGdnConfig{16, 48, 128, 128, 4};
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

#ifdef QW38_DIAGNOSTIC_TRACE
struct TraceContext final {
  const internal::TraceFilter* filter = nullptr;
  internal::TraceSink sink = nullptr;
  void* context = nullptr;
};
thread_local TraceContext* active_trace = nullptr;

Status emit_cuda_trace(const internal::TraceFilter& filter,
                       internal::TraceSink sink, void* context,
                       const char* name, std::size_t layer,
                       const float* device_values, std::size_t count,
                       float* host_values) noexcept {
  cudaError_t error = cudaMemcpy(host_values, device_values,
                                 count * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return cuda_status(error, "cannot copy CUDA trace tensor");
  internal::TraceTensorView view{name, layer, host_values, count,
                                 {count, 0, 0}, 1};
  return internal::emit_trace_tensor(filter, sink, context, view);
}
#endif

class NvtxRange final {
 public:
  explicit NvtxRange(const char* name) noexcept { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

class GpuPhaseRecorder final {
 public:
  GpuPhaseRecorder() noexcept = default;
  ~GpuPhaseRecorder() {
    const std::size_t cleanup_count = count_ + (active_ ? 1 : 0);
    for (std::size_t index = 0; index < cleanup_count; ++index) {
      if (phases_[index].stop != nullptr) cudaEventDestroy(phases_[index].stop);
      if (phases_[index].start != nullptr) {
        cudaEventDestroy(phases_[index].start);
      }
    }
  }
  GpuPhaseRecorder(const GpuPhaseRecorder&) = delete;
  GpuPhaseRecorder& operator=(const GpuPhaseRecorder&) = delete;

  cudaError_t begin(TimingValue* destination, cudaStream_t stream) noexcept {
    if (destination == nullptr || active_ || count_ == phases_.size()) {
      return cudaErrorInvalidValue;
    }
    Phase& phase = phases_[count_];
    phase.destination = destination;
    phase.stream = stream;
    cudaError_t error = cudaEventCreate(&phase.start);
    if (error == cudaSuccess) error = cudaEventCreate(&phase.stop);
    if (error == cudaSuccess) error = cudaEventRecord(phase.start, stream);
    if (error != cudaSuccess) {
      if (phase.stop != nullptr) cudaEventDestroy(phase.stop);
      if (phase.start != nullptr) cudaEventDestroy(phase.start);
      phase = {};
      return error;
    }
    active_ = true;
    return cudaSuccess;
  }

  cudaError_t end() noexcept {
    if (!active_) return cudaErrorInvalidValue;
    Phase& phase = phases_[count_];
    cudaError_t error = cudaEventRecord(phase.stop, phase.stream);
    if (error == cudaSuccess) {
      ++count_;
      active_ = false;
    }
    return error;
  }

  cudaError_t collect() noexcept {
    if (active_) return cudaErrorInvalidValue;
    for (std::size_t index = 0; index < count_; ++index) {
      Phase& phase = phases_[index];
      cudaError_t error = cudaEventSynchronize(phase.stop);
      float milliseconds = 0.0F;
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&milliseconds, phase.start, phase.stop);
      }
      if (error != cudaSuccess) return error;
      phase.destination->milliseconds += milliseconds;
      phase.destination->measured = true;
    }
    return cudaSuccess;
  }

  cudaError_t reset() noexcept {
    if (active_) return cudaErrorInvalidValue;
    for (std::size_t index = 0; index < count_; ++index) {
      if (phases_[index].stop != nullptr) cudaEventDestroy(phases_[index].stop);
      if (phases_[index].start != nullptr) {
        cudaEventDestroy(phases_[index].start);
      }
      phases_[index] = {};
    }
    count_ = 0;
    active_ = false;
    return cudaSuccess;
  }

 private:
  struct Phase final {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaStream_t stream = nullptr;
    TimingValue* destination = nullptr;
  };
  // Enclosing OPT-038 categories plus OPT-043 exclusive leaves. Decode with
  // per-projection leaves needs ~16 phases per layer.
  std::array<Phase, 1536> phases_{};
  std::size_t count_ = 0;
  bool active_ = false;
};

cudaError_t begin_phase(GpuPhaseRecorder* recorder, TimingValue* destination,
                        cudaStream_t stream = nullptr) noexcept {
  return recorder == nullptr ? cudaSuccess
                             : recorder->begin(destination, stream);
}

cudaError_t end_phase(GpuPhaseRecorder* recorder) noexcept {
  return recorder == nullptr ? cudaSuccess : recorder->end();
}

int capture_slot_index(const ActivationCapture* capture,
                       std::size_t layer) noexcept {
  if (capture == nullptr) return -1;
  for (std::size_t index = 0; index < capture->layers.size(); ++index) {
    if (capture->layers[index] == layer) return static_cast<int>(index);
  }
  return -1;
}

void write_float_sha(const float* values, std::size_t count, char* hex) noexcept {
  hex[0] = '\0';
  if (values == nullptr || count == 0) return;
  std::string digest;
  const Status status = internal::sha256_bytes(
      reinterpret_cast<const unsigned char*>(values), count * sizeof(float),
      &digest);
  if (!status.is_ok() || digest.size() != 64) return;
  std::memcpy(hex, digest.data(), 64);
  hex[64] = '\0';
}

void copy_prefix(const float* values, std::size_t count, float* prefix,
                 std::size_t prefix_count) noexcept {
  const std::size_t n = std::min(count, prefix_count);
  for (std::size_t index = 0; index < n; ++index) prefix[index] = values[index];
}

cudaError_t capture_bf16_vector(const __nv_bfloat16* device, std::size_t count,
                                float* host, char* sha, float* prefix,
                                std::size_t prefix_count) noexcept {
  if (device == nullptr || host == nullptr || count == 0 ||
      count > internal::kResidualWidth) {
    return cudaErrorInvalidValue;
  }
  std::array<__nv_bfloat16, internal::kResidualWidth> tmp{};
  cudaError_t error = cudaMemcpy(tmp.data(), device,
                                 count * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return error;
  for (std::size_t index = 0; index < count; ++index) {
    host[index] = __bfloat162float(tmp[index]);
  }
  write_float_sha(host, count, sha);
  copy_prefix(host, count, prefix, prefix_count);
  return cudaSuccess;
}

cudaError_t capture_fp32_vector(const float* device, std::size_t count,
                                float* host, char* sha, float* prefix,
                                std::size_t prefix_count) noexcept {
  if (device == nullptr || host == nullptr || count == 0) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error =
      cudaMemcpy(host, device, count * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return error;
  write_float_sha(host, count, sha);
  copy_prefix(host, count, prefix, prefix_count);
  return cudaSuccess;
}

cudaError_t maybe_capture_preprojection(ActivationCapture* capture,
                                        std::size_t layer, bool mixer,
                                        const char* layer_kind,
                                        const __nv_bfloat16* device,
                                        std::size_t count) noexcept {
  const int slot = capture_slot_index(capture, layer);
  if (slot < 0) return cudaSuccess;
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return error;
  ActivationCaptureSlot& dest = capture->slots[static_cast<std::size_t>(slot)];
  dest.layer = layer;
  dest.layer_kind = layer_kind;
  if (mixer) {
    std::snprintf(dest.mixer_dtype, sizeof(dest.mixer_dtype), "BF16");
    dest.mixer_shape = {1, count};
    error = capture_bf16_vector(
        device, count, dest.mixer_preprojection.data(), dest.mixer_sha256,
        dest.mixer_prefix.data(), dest.mixer_prefix.size());
    if (error == cudaSuccess) dest.mixer_captured = true;
    return error;
  }
  std::snprintf(dest.ffn_dtype, sizeof(dest.ffn_dtype), "BF16");
  dest.ffn_shape = {1, count};
  error = capture_bf16_vector(
      device, count, dest.ffn_preprojection.data(), dest.ffn_sha256,
      dest.ffn_prefix.data(), dest.ffn_prefix.size());
  if (error == cudaSuccess) dest.ffn_captured = true;
  return error;
}

void copy_graph_host_leaf(LeafTimings* leaves, const TimingValue& graph) noexcept {
  if (leaves == nullptr) return;
  leaves->host_graph_submit = graph;
}

void reconstruct_decode_leaves(DecodeAttribution* attribution) noexcept {
  if (attribution == nullptr || !attribution->record_leaves) return;
  copy_graph_host_leaf(&attribution->leaves, attribution->graph);
}

void reconstruct_prefill_leaves(PrefillAttribution* attribution) noexcept {
  if (attribution == nullptr || !attribution->record_leaves) return;
  copy_graph_host_leaf(&attribution->leaves, attribution->graph);
}

__device__ float read_q8_half(const std::uint8_t* bytes) {
  const unsigned short bits = static_cast<unsigned short>(bytes[0]) |
                              (static_cast<unsigned short>(bytes[1]) << 8U);
  return __half2float(__ushort_as_half(bits));
}

__global__ void q8_mmv_bf16(const std::uint8_t* weights, std::size_t rows,
                            std::size_t columns,
                            const __nv_bfloat16* activation, float* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * (kThreads / kWarpSize) + warp;
  if (row >= rows) return;
  const std::uint8_t* row_weights = weights + row * (columns / 32) * 34;
  float sum = 0.0F;
  for (std::size_t column = lane; column < columns; column += kWarpSize) {
    const std::uint8_t* block = row_weights + (column / 32) * 34;
    const float weight =
        read_q8_half(block) *
        static_cast<float>(static_cast<std::int8_t>(block[2 + column % 32]));
    sum = __fadd_rn(sum,
                    __fmul_rn(weight, __bfloat162float(activation[column])));
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sum = __fadd_rn(
        sum, __shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarpSize));
  }
  if (lane == 0) output[row] = sum;
}

__global__ void bf16_to_fp32(const __nv_bfloat16* input, std::size_t count,
                             float* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = __bfloat162float(input[index]);
}

__global__ void rms_norm_fp32_to_bf16(const float* input, const float* scale,
                                      std::size_t count,
                                      __nv_bfloat16* output) {
  quartz_pdl_sync();
  __shared__ float inverse;
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
      sum = __fadd_rn(sum, __fmul_rn(input[index], input[index]));
    }
    inverse = 1.0F / sqrtf(sum / static_cast<float>(count) + 1.0e-6F);
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    output[index] =
        __float2bfloat16_rn(__fmul_rn(__fmul_rn(input[index], inverse),
                                     scale[index]));
  }
  quartz_pdl_lc();
}

__global__ void rms_norm_rows_fp32_to_bf16(
    const float* input, const float* scale, std::size_t width,
    __nv_bfloat16* output) {
  quartz_pdl_sync();
  const std::size_t row = blockIdx.x;
  const float* row_input = input + row * width;
  __nv_bfloat16* row_output = output + row * width;
  __shared__ float inverse;
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < width; ++index) {
      sum = __fadd_rn(sum,
                      __fmul_rn(row_input[index], row_input[index]));
    }
    inverse = 1.0F / sqrtf(sum / static_cast<float>(width) + 1.0e-6F);
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    row_output[index] = __float2bfloat16_rn(
        __fmul_rn(__fmul_rn(row_input[index], inverse), scale[index]));
  }
  quartz_pdl_lc();
}

template <bool UseRsqrt>
__global__ void rms_norm_fp32_to_bf16_parallel(const float* input,
                                               const float* scale,
                                               std::size_t count,
                                               __nv_bfloat16* output) {
  quartz_pdl_sync();
  const float inverse = cooperative_rms_inverse_fp32<UseRsqrt>(input, count);
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    output[index] = rms_norm_store_bf16(input[index], inverse, scale[index]);
  }
  quartz_pdl_lc();
}

template <bool UseRsqrt>
__global__ void rms_norm_rows_fp32_to_bf16_parallel(
    const float* input, const float* scale, std::size_t width,
    __nv_bfloat16* output) {
  quartz_pdl_sync();
  const std::size_t row = blockIdx.x;
  const float* row_input = input + row * width;
  __nv_bfloat16* row_output = output + row * width;
  const float inverse =
      cooperative_rms_inverse_fp32<UseRsqrt>(row_input, width);
  for (std::size_t index = threadIdx.x; index < width; index += blockDim.x) {
    row_output[index] =
        rms_norm_store_bf16(row_input[index], inverse, scale[index]);
  }
  quartz_pdl_lc();
}

__global__ void residual_add_fp32(const float* residual,
                                  const float* correction,
                                  std::size_t count, float* output) {
  quartz_pdl_sync();
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __fadd_rn(residual[index], correction[index]);
  }
  quartz_pdl_lc();
}

__global__ void prepare_gdn_gate_rows(
    const float* alpha, const float* beta, const float* folded_a,
    const float* dt_bias, std::size_t key_heads, std::size_t replicas,
    float* log_decay, float* update) {
  quartz_pdl_sync();
  const std::size_t row = blockIdx.x;
  const std::size_t grouped = threadIdx.x;
  const std::size_t count = key_heads * replicas;
  if (grouped >= count) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t key = grouped / replicas;
  const std::size_t replica = grouped % replicas;
  const std::size_t tiled = replica * key_heads + key;
  const float gate = alpha[row * count + tiled] + dt_bias[tiled];
  const float softplus =
      gate > 20.0F ? gate : gate < -20.0F ? expf(gate) : log1pf(expf(gate));
  log_decay[row * count + grouped] = __fmul_rn(folded_a[tiled], softplus);
  const float beta_value = beta[row * count + tiled];
  update[row * count + grouped] =
      beta_value >= 0.0F
          ? 1.0F / (1.0F + expf(-beta_value))
          : expf(beta_value) / (1.0F + expf(beta_value));
  quartz_pdl_lc();
}

__global__ void gdn_gated_output_rows(
    const float* recurrent, const float* gate_tiled, const float* norm,
    std::size_t key_heads, std::size_t replicas, std::size_t head_width,
    __nv_bfloat16* output_tiled) {
  quartz_pdl_sync();
  const std::size_t row = blockIdx.y;
  const std::size_t grouped_head = blockIdx.x;
  const std::size_t lane = threadIdx.x;
  const std::size_t heads = key_heads * replicas;
  const std::size_t grouped_base =
      (row * heads + grouped_head) * head_width;
  __shared__ float inverse;
  if (lane == 0) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < head_width; ++index) {
      const float value = recurrent[grouped_base + index];
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    inverse = 1.0F /
              sqrtf(sum / static_cast<float>(head_width) + 1.0e-6F);
  }
  __syncthreads();
  if (lane < head_width) {
    const std::size_t key = grouped_head / replicas;
    const std::size_t replica = grouped_head % replicas;
    const std::size_t tiled_head = replica * key_heads + key;
    const std::size_t tiled_base =
        (row * heads + tiled_head) * head_width;
    const float gate = gate_tiled[tiled_base + lane];
    const float silu = gate / (1.0F + expf(-gate));
    const float value = __fmul_rn(
        __fmul_rn(recurrent[grouped_base + lane], inverse), norm[lane]);
    output_tiled[tiled_base + lane] =
        __float2bfloat16_rn(__fmul_rn(value, silu));
  }
  quartz_pdl_lc();
}

template <bool UseRsqrt>
__global__ void gdn_gated_output_rows_parallel(
    const float* recurrent, const float* gate_tiled, const float* norm,
    std::size_t key_heads, std::size_t replicas, std::size_t head_width,
    __nv_bfloat16* output_tiled) {
  quartz_pdl_sync();
  const std::size_t row = blockIdx.y;
  const std::size_t grouped_head = blockIdx.x;
  const std::size_t heads = key_heads * replicas;
  const std::size_t grouped_base =
      (row * heads + grouped_head) * head_width;
  const float inverse = cooperative_rms_inverse_fp32<UseRsqrt>(
      recurrent + grouped_base, head_width);
  const std::size_t key = grouped_head / replicas;
  const std::size_t replica = grouped_head % replicas;
  const std::size_t tiled_head = replica * key_heads + key;
  const std::size_t tiled_base = (row * heads + tiled_head) * head_width;
  for (std::size_t lane = threadIdx.x; lane < head_width;
       lane += blockDim.x) {
    const float gate = gate_tiled[tiled_base + lane];
    const float silu = gate / (1.0F + expf(-gate));
    const float value = __fmul_rn(
        __fmul_rn(recurrent[grouped_base + lane], inverse), norm[lane]);
    output_tiled[tiled_base + lane] =
        __float2bfloat16_rn(__fmul_rn(value, silu));
  }
  quartz_pdl_lc();
}

__global__ void split_attention_rows(const float* packed,
                                     std::size_t query_values,
                                     std::size_t head_width,
                                     std::size_t token_count, float* query,
                                     float* gate) {
  quartz_pdl_sync();
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = token_count * query_values;
  if (index >= count) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t row = index / query_values;
  const std::size_t within = index % query_values;
  const std::size_t head = within / head_width;
  const std::size_t lane = within % head_width;
  const std::size_t packed_base =
      row * query_values * 2 + head * head_width * 2;
  query[index] = packed[packed_base + lane];
  gate[index] = packed[packed_base + head_width + lane];
  quartz_pdl_lc();
}

__global__ void residual_add_norm_fp32_to_bf16(
    const float* residual, const float* correction, const float* scale,
    std::size_t count, float* output, __nv_bfloat16* normalized) {
  __shared__ float inverse;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    output[index] = __fadd_rn(residual[index], correction[index]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
      const float value = output[index];
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    inverse = 1.0F / sqrtf(sum / static_cast<float>(count) + 1.0e-6F);
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    normalized[index] = __float2bfloat16_rn(
        __fmul_rn(__fmul_rn(output[index], inverse), scale[index]));
  }
}

__global__ void residual_add_norm_rows_fp32_to_bf16(
    const float* residual, const float* correction, const float* scale,
    std::size_t width, float* output, __nv_bfloat16* normalized) {
  const std::size_t row = blockIdx.x;
  const float* row_residual = residual + row * width;
  const float* row_correction = correction + row * width;
  float* row_output = output + row * width;
  __nv_bfloat16* row_normalized = normalized + row * width;
  __shared__ float inverse;
  for (std::size_t index = threadIdx.x; index < width; index += blockDim.x) {
    row_output[index] = __fadd_rn(row_residual[index], row_correction[index]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < width; ++index) {
      const float value = row_output[index];
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    inverse = 1.0F / sqrtf(sum / static_cast<float>(width) + 1.0e-6F);
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < width; index += blockDim.x) {
    row_normalized[index] = __float2bfloat16_rn(
        __fmul_rn(__fmul_rn(row_output[index], inverse), scale[index]));
  }
}

template <bool UseRsqrt>
__global__ void residual_add_norm_fp32_to_bf16_parallel(
    const float* residual, const float* correction, const float* scale,
    std::size_t count, float* output, __nv_bfloat16* normalized) {
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    output[index] = __fadd_rn(residual[index], correction[index]);
  }
  __syncthreads();
  const float inverse = cooperative_rms_inverse_fp32<UseRsqrt>(output, count);
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    normalized[index] =
        rms_norm_store_bf16(output[index], inverse, scale[index]);
  }
}

template <bool UseRsqrt>
__global__ void residual_add_norm_rows_fp32_to_bf16_parallel(
    const float* residual, const float* correction, const float* scale,
    std::size_t width, float* output, __nv_bfloat16* normalized) {
  const std::size_t row = blockIdx.x;
  const float* row_residual = residual + row * width;
  const float* row_correction = correction + row * width;
  float* row_output = output + row * width;
  __nv_bfloat16* row_normalized = normalized + row * width;
  for (std::size_t index = threadIdx.x; index < width; index += blockDim.x) {
    row_output[index] = __fadd_rn(row_residual[index], row_correction[index]);
  }
  __syncthreads();
  const float inverse = cooperative_rms_inverse_fp32<UseRsqrt>(row_output, width);
  for (std::size_t index = threadIdx.x; index < width; index += blockDim.x) {
    row_normalized[index] =
        rms_norm_store_bf16(row_output[index], inverse, scale[index]);
  }
}

__global__ void compare_bytes(const std::uint8_t* left,
                              const std::uint8_t* right,
                              std::size_t count,
                              unsigned int* mismatch) {
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    if (left[index] != right[index]) atomicExch(mismatch, 1U);
  }
}

Status cuda_status(cudaError_t error, const char* message) noexcept {
  return error == cudaSuccess
             ? Status::ok()
             : Status{StatusCode::kInternal, message};
}

int rms_norm_block_threads() noexcept {
  const int threads = effective_rms_norm_threads();
  return legal_rms_norm_threads(threads) ? threads : kThreads;
}

int gdn_norm_block_threads() noexcept {
  const int threads = effective_gdn_norm_threads();
  return legal_rms_norm_threads(threads) ? threads : kThreads;
}

cudaError_t launch_rms_norm_fp32_dispatch(const float* input, const float* scale,
                                          std::size_t count,
                                          __nv_bfloat16* output,
                                          cudaStream_t stream) noexcept {
  if (rms_norm_uses_serial()) {
    rms_norm_fp32_to_bf16<<<1, kThreads, 0, stream>>>(input, scale, count,
                                                      output);
    return cudaPeekAtLastError();
  }
  const dim3 block(static_cast<unsigned int>(rms_norm_block_threads()));
  if (rms_norm_uses_rsqrt()) {
    if (block.x == 128) {
      return quartz_launch_kernel(rms_norm_fp32_to_bf16_parallel<true>, dim3(1),
                                  block, 0, stream, input, scale, count,
                                  output);
    }
    return quartz_launch_kernel(rms_norm_fp32_to_bf16_parallel<true>, dim3(1),
                                block, 0, stream, input, scale, count, output);
  }
  if (block.x == 128) {
    return quartz_launch_kernel(rms_norm_fp32_to_bf16_parallel<false>, dim3(1),
                                block, 0, stream, input, scale, count, output);
  }
  return quartz_launch_kernel(rms_norm_fp32_to_bf16_parallel<false>, dim3(1),
                              block, 0, stream, input, scale, count, output);
}

cudaError_t launch_rms_norm_rows_dispatch(const float* input, const float* scale,
                                          std::size_t width,
                                          std::size_t token_count,
                                          __nv_bfloat16* output,
                                          cudaStream_t stream) noexcept {
  const dim3 grid(static_cast<unsigned int>(token_count));
  if (rms_norm_uses_serial()) {
    return quartz_launch_kernel(rms_norm_rows_fp32_to_bf16, grid, dim3(kThreads),
                                0, stream, input, scale, width, output);
  }
  const dim3 block(static_cast<unsigned int>(rms_norm_block_threads()));
  if (rms_norm_uses_rsqrt()) {
    return quartz_launch_kernel(rms_norm_rows_fp32_to_bf16_parallel<true>, grid,
                                block, 0, stream, input, scale, width, output);
  }
  return quartz_launch_kernel(rms_norm_rows_fp32_to_bf16_parallel<false>, grid,
                              block, 0, stream, input, scale, width, output);
}

cudaError_t launch_residual_add_norm_fp32_dispatch(
    const float* residual, const float* correction, const float* scale,
    std::size_t count, float* output, __nv_bfloat16* normalized,
    cudaStream_t stream) noexcept {
  if (rms_norm_uses_serial()) {
    residual_add_norm_fp32_to_bf16<<<1, kThreads, 0, stream>>>(
        residual, correction, scale, count, output, normalized);
    return cudaPeekAtLastError();
  }
  const dim3 block(static_cast<unsigned int>(rms_norm_block_threads()));
  if (rms_norm_uses_rsqrt()) {
    return quartz_launch_kernel(residual_add_norm_fp32_to_bf16_parallel<true>,
                                dim3(1), block, 0, stream, residual, correction,
                                scale, count, output, normalized);
  }
  return quartz_launch_kernel(residual_add_norm_fp32_to_bf16_parallel<false>,
                              dim3(1), block, 0, stream, residual, correction,
                              scale, count, output, normalized);
}

cudaError_t launch_residual_add_norm_rows_dispatch(
    const float* residual, const float* correction, const float* scale,
    std::size_t width, std::size_t token_count, float* output,
    __nv_bfloat16* normalized, cudaStream_t stream) noexcept {
  const dim3 grid(static_cast<unsigned int>(token_count));
  if (rms_norm_uses_serial()) {
    residual_add_norm_rows_fp32_to_bf16<<<static_cast<unsigned int>(token_count),
                                          kThreads, 0, stream>>>(
        residual, correction, scale, width, output, normalized);
    return cudaPeekAtLastError();
  }
  const dim3 block(static_cast<unsigned int>(rms_norm_block_threads()));
  if (rms_norm_uses_rsqrt()) {
    return quartz_launch_kernel(
        residual_add_norm_rows_fp32_to_bf16_parallel<true>, grid, block, 0,
        stream, residual, correction, scale, width, output, normalized);
  }
  return quartz_launch_kernel(
      residual_add_norm_rows_fp32_to_bf16_parallel<false>, grid, block, 0,
      stream, residual, correction, scale, width, output, normalized);
}

cudaError_t launch_gdn_gated_output_rows_dispatch(
    const float* recurrent, const float* gate_tiled, const float* norm,
    std::size_t key_heads, std::size_t replicas, std::size_t head_width,
    std::size_t token_count, __nv_bfloat16* output_tiled,
    cudaStream_t stream) noexcept {
  const dim3 grid(static_cast<unsigned int>(key_heads * replicas),
                  static_cast<unsigned int>(token_count));
  if (rms_norm_uses_serial()) {
    return quartz_launch_kernel(gdn_gated_output_rows, grid, dim3(kThreads), 0,
                                stream, recurrent, gate_tiled, norm, key_heads,
                                replicas, head_width, output_tiled);
  }
  const dim3 block(static_cast<unsigned int>(gdn_norm_block_threads()));
  if (rms_norm_uses_rsqrt()) {
    return quartz_launch_kernel(gdn_gated_output_rows_parallel<true>, grid,
                                block, 0, stream, recurrent, gate_tiled, norm,
                                key_heads, replicas, head_width, output_tiled);
  }
  return quartz_launch_kernel(gdn_gated_output_rows_parallel<false>, grid, block,
                              0, stream, recurrent, gate_tiled, norm,
                              key_heads, replicas, head_width, output_tiled);
}

constexpr float kPrefillAbsTolMs = 0.05F;
constexpr float kPrefillRelTol = 1.0e-4F;

bool prefill_timings_close(float left, float right) noexcept {
  const float difference = std::fabs(left - right);
  return difference <= std::max(kPrefillRelTol * std::max(std::fabs(left),
                                                            std::fabs(right)),
                                 kPrefillAbsTolMs);
}

Status finish_prefill_attribution(
    PrefillAttribution* attribution,
    std::chrono::steady_clock::time_point started) noexcept {
  if (attribution == nullptr) return Status::ok();
  attribution->wall = {
      static_cast<float>(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count()),
      true};
  attribution->graph.measured = true;
  const TimingValue* required[] = {
      &attribution->embedding,       &attribution->mixer_mmq,
      &attribution->gdn_core,        &attribution->attention_core,
      &attribution->ffn_mmq,         &attribution->logits,
      &attribution->commit_sync,     &attribution->graph,
      &attribution->wall};
  for (const TimingValue* value : required) {
    if (!value->measured) {
      return {StatusCode::kInternal,
              "CUDA prefill attribution left a category unmeasured"};
    }
  }
  const float exclusive = attribution->embedding.milliseconds +
                           attribution->mixer_mmq.milliseconds +
                           attribution->gdn_core.milliseconds +
                           attribution->attention_core.milliseconds +
                           attribution->ffn_mmq.milliseconds +
                           attribution->logits.milliseconds +
                           attribution->commit_sync.milliseconds +
                           attribution->graph.milliseconds;
  if (exclusive > attribution->wall.milliseconds &&
      !prefill_timings_close(exclusive, attribution->wall.milliseconds)) {
    if (!attribution->record_leaves) {
      return {StatusCode::kInternal,
              "CUDA prefill attribution exclusive sum exceeds wall time"};
    }
    attribution->wall.milliseconds = exclusive;
  }
  attribution->other_idle = {
      std::max(0.0F, attribution->wall.milliseconds - exclusive), true};
  reconstruct_prefill_leaves(attribution);
  return Status::ok();
}

Status finish_decode_attribution(
    DecodeAttribution* attribution,
    std::chrono::steady_clock::time_point started) noexcept {
  if (attribution == nullptr) return Status::ok();
  attribution->wall = {
      static_cast<float>(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count()),
      true};
  attribution->graph.measured = true;
  const TimingValue* required[] = {
      &attribution->embedding,       &attribution->mixer_mmv,
      &attribution->gdn_core,        &attribution->attention_core,
      &attribution->ffn_mmv,         &attribution->logits,
      &attribution->state_commit,    &attribution->graph,
      &attribution->wall};
  for (const TimingValue* value : required) {
    if (!value->measured) {
      return {StatusCode::kInternal,
              "CUDA decode attribution left a category unmeasured"};
    }
  }
  const float cuda_sum = attribution->embedding.milliseconds +
                          attribution->mixer_mmv.milliseconds +
                          attribution->gdn_core.milliseconds +
                          attribution->attention_core.milliseconds +
                          attribution->ffn_mmv.milliseconds +
                          attribution->logits.milliseconds +
                          attribution->state_commit.milliseconds;
  const float exclusive = cuda_sum + attribution->graph.milliseconds;
  if (cuda_sum > attribution->wall.milliseconds &&
      !prefill_timings_close(cuda_sum, attribution->wall.milliseconds)) {
    return {StatusCode::kInternal,
            "CUDA decode attribution exclusive sum exceeds wall time"};
  }
  if (exclusive > attribution->wall.milliseconds) {
    attribution->wall.milliseconds = exclusive;
  }
  attribution->other_idle = {
      std::max(0.0F, attribution->wall.milliseconds - exclusive), true};
  reconstruct_decode_leaves(attribution);
  return Status::ok();
}

template <typename T>
cudaError_t allocate(T** pointer, std::size_t count,
                     std::size_t* total) noexcept {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    return cudaErrorInvalidValue;
  }
  const std::size_t bytes = count * sizeof(T);
  const cudaError_t error = cudaMalloc(pointer, bytes);
  if (error == cudaSuccess) *total += bytes;
  return error;
}

bool remap_bytes(const std::uint8_t* host, std::size_t bytes,
                 const std::uint8_t* base, std::size_t base_bytes,
                 const std::uint8_t* device_base,
                 const std::uint8_t** device) noexcept {
  if (host == nullptr || base == nullptr || device == nullptr || host < base) {
    return false;
  }
  const std::size_t offset = static_cast<std::size_t>(host - base);
  if (offset > base_bytes || bytes > base_bytes - offset) return false;
  *device = device_base + offset;
  return true;
}

bool remap_vector(const internal::VectorView& source,
                  const std::uint8_t* base, std::size_t base_bytes,
                  const std::uint8_t* device_base,
                  const float** destination) noexcept {
  const std::uint8_t* bytes = nullptr;
  if (source.type != 0 ||
      !remap_bytes(source.data, source.storage_bytes, base, base_bytes,
                   device_base, &bytes)) {
    return false;
  }
  *destination = reinterpret_cast<const float*>(bytes);
  return true;
}

bool remap_tensor(const internal::TensorView& source,
                  const std::uint8_t* base, std::size_t base_bytes,
                  const std::uint8_t* device_base,
                  DeviceTensor* destination) noexcept {
  const std::uint8_t* bytes = nullptr;
  QuantKind kind = QuantKind::kQ4K;
  if (source.type == 8) {
    kind = QuantKind::kQ8_0;
  } else if (source.type == 12) {
    kind = QuantKind::kQ4K;
  } else if (source.type == 14) {
    kind = QuantKind::kQ6K;
  } else {
    return false;
  }
  if (!remap_bytes(source.data, source.storage_bytes, base, base_bytes,
                   device_base, &bytes)) {
    return false;
  }
  *destination = {bytes, source.columns, source.rows, kind};
  return true;
}

bool remap_common(const internal::CommonLayerWeights& source,
                  const std::uint8_t* base, std::size_t bytes,
                  const std::uint8_t* device, DeviceCommonLayer* output) {
  return remap_vector(source.input_norm, base, bytes, device,
                      &output->input_norm) &&
         remap_tensor(source.ffn_gate, base, bytes, device,
                      &output->ffn_gate) &&
         remap_tensor(source.ffn_up, base, bytes, device, &output->ffn_up) &&
         remap_tensor(source.ffn_down, base, bytes, device,
                      &output->ffn_down) &&
         remap_vector(source.ffn_norm, base, bytes, device,
                      &output->ffn_norm);
}

cudaError_t matrix_vector(const DeviceTensor& matrix,
                          const __nv_bfloat16* activation,
                          SchedulerWorkspace* workspace,
                          float* output, cudaStream_t stream) noexcept {
  if (matrix.kind == QuantKind::kQ8_0) {
    if (q8_decode_uses_dp4a_for_rows(matrix.rows) &&
        matrix.columns % 32 == 0 && workspace != nullptr &&
        workspace->q8_ != nullptr) {
      cudaError_t error = cudaSuccess;
      if (workspace->q8_decode_staged_activation_ != activation ||
          workspace->q8_decode_staged_columns_ != matrix.columns) {
        error = launch_quantize_bf16_q8_1(activation, workspace->q8_,
                                           matrix.columns, stream);
        if (error != cudaSuccess) return error;
        workspace->q8_decode_staged_activation_ = activation;
        workspace->q8_decode_staged_columns_ = matrix.columns;
      }
      return launch_q8_coop_mmv_prequant(
          matrix.data, matrix.rows, matrix.columns, workspace->q8_, output,
          q8_decode_warps_for_rows(matrix.rows), stream);
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (matrix.rows + (kThreads / kWarpSize) - 1) /
        (kThreads / kWarpSize));
    return quartz_launch_kernel(
        q8_mmv_bf16, dim3(blocks), dim3(kThreads), 0, stream, matrix.data,
        matrix.rows, matrix.columns, activation, output);
  }
  if (workspace != nullptr) {
    workspace->q8_decode_staged_activation_ = nullptr;
    workspace->q8_decode_staged_columns_ = 0;
  }
  return launch_quant_mmv(matrix.kind, matrix.data, matrix.rows,
                          matrix.columns, activation, workspace->q8_, output,
                          stream);
}

cudaError_t matrix_prompt(const DeviceTensor& matrix,
                          const __nv_bfloat16* activation,
                          std::size_t prompt_rows,
                          SchedulerWorkspace* workspace, float* output,
                          cudaStream_t stream) noexcept {
  if (matrix.kind == QuantKind::kQ8_0) {
    return launch_q8_mmq_bf16(matrix.data, matrix.rows, matrix.columns,
                              activation, prompt_rows, workspace->prompt_q8_,
                              output, stream);
  }
  const std::size_t fixup_floats =
      workspace->prompt_chunk_rows_ * kMaximumProjection;
  return launch_quant_mmq(matrix.kind, matrix.data, matrix.rows,
                          matrix.columns, activation, prompt_rows,
                          workspace->prompt_q8_, output, stream,
                          workspace->prompt_projection_a_, fixup_floats);
}

cudaError_t matrix_prompt_q8_quality_mma(const DeviceTensor& matrix,
                                         const Q8Block* y,
                                         std::size_t prompt_rows, float* output,
                                         cudaStream_t stream) noexcept {
  return launch_q8_mmq_quality_mma(matrix.data, matrix.rows, matrix.columns, y,
                                   prompt_rows, output, stream);
}

cudaError_t execute_prompt_ffn_projections(
    const DeviceCommonLayer& layer, SchedulerWorkspace* workspace,
    std::size_t token_count, cudaStream_t stream,
    GpuPhaseRecorder* leaves = nullptr,
    LeafTimings* leaf_timings = nullptr) noexcept {
  cudaError_t error = cudaSuccess;
  const bool share_y = ffn_shares_gate_up_y() &&
                       layer.ffn_gate.kind == QuantKind::kQ4K &&
                       layer.ffn_up.kind == QuantKind::kQ4K &&
                       layer.ffn_gate.columns == internal::kResidualWidth &&
                       layer.ffn_up.columns == internal::kResidualWidth;
  const bool swiglu_q8 = ffn_swiglu_writes_q8() &&
                         layer.ffn_down.kind == QuantKind::kQ4K &&
                         layer.ffn_down.columns == internal::kFfnWidth;
  if (leaf_timings != nullptr) {
    leaf_timings->swiglu_fused_with_down_stage = swiglu_q8;
  }
  if (share_y) {
    error = begin_phase(leaves,
                        leaf_timings == nullptr
                            ? nullptr
                            : &leaf_timings->activation_staging_ffn,
                        stream);
    if (error == cudaSuccess) {
      error = launch_quantize_mmq_q8_1(
          QuantKind::kQ4K, workspace->prompt_normalized_, token_count,
          internal::kResidualWidth, workspace->prompt_q8_, stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) {
      error = begin_phase(leaves,
                          leaf_timings == nullptr
                              ? nullptr
                              : &leaf_timings->proj_ffn_gate,
                          stream);
    }
    if (error == cudaSuccess) {
      error = launch_quant_mmq_mma_y_ij(
          layer.ffn_gate.kind, layer.ffn_gate.data, layer.ffn_gate.rows,
          layer.ffn_gate.columns, workspace->prompt_q8_, token_count,
          workspace->prompt_projection_a_, selected_ffn_gate_quality_i(),
          selected_ffn_gate_prompt_tile(), stream,
          reinterpret_cast<float*>(workspace->prompt_projected_bf16_),
          workspace->prompt_chunk_rows_ * internal::kFfnWidth *
              sizeof(__nv_bfloat16) / sizeof(float));
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) {
      error = begin_phase(
          leaves,
          leaf_timings == nullptr ? nullptr : &leaf_timings->proj_ffn_up,
          stream);
    }
    if (error == cudaSuccess) {
      error = launch_quant_mmq_mma_y_ij(
          layer.ffn_up.kind, layer.ffn_up.data, layer.ffn_up.rows,
          layer.ffn_up.columns, workspace->prompt_q8_, token_count,
          workspace->prompt_projection_b_, selected_ffn_up_quality_i(),
          selected_ffn_up_prompt_tile(), stream,
          reinterpret_cast<float*>(workspace->prompt_projected_bf16_),
          workspace->prompt_chunk_rows_ * internal::kFfnWidth *
              sizeof(__nv_bfloat16) / sizeof(float));
    }
    if (error == cudaSuccess) error = end_phase(leaves);
  } else {
    if (leaf_timings != nullptr) leaf_timings->ffn_staging_fused_into_mmv = true;
    error = begin_phase(
        leaves,
        leaf_timings == nullptr ? nullptr : &leaf_timings->proj_ffn_gate,
        stream);
    if (error == cudaSuccess) {
      error = matrix_prompt(layer.ffn_gate, workspace->prompt_normalized_,
                            token_count, workspace,
                            workspace->prompt_projection_a_, stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) {
      error = begin_phase(
          leaves,
          leaf_timings == nullptr ? nullptr : &leaf_timings->proj_ffn_up,
          stream);
    }
    if (error == cudaSuccess) {
      error = matrix_prompt(layer.ffn_up, workspace->prompt_normalized_,
                            token_count, workspace,
                            workspace->prompt_projection_b_, stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
  }
  if (error == cudaSuccess) {
    if (swiglu_q8) {
      error = begin_phase(
          leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->swiglu,
          stream);
      if (error == cudaSuccess) {
        error = launch_swiglu_quantize_mmq_q8_1(
            workspace->prompt_projection_a_, workspace->prompt_projection_b_,
            token_count, internal::kFfnWidth, workspace->prompt_q8_, stream);
      }
      if (error == cudaSuccess) error = end_phase(leaves);
      if (error == cudaSuccess) {
        error = begin_phase(leaves,
                            leaf_timings == nullptr
                                ? nullptr
                                : &leaf_timings->proj_ffn_down,
                            stream);
      }
      if (error == cudaSuccess) {
        error = launch_quant_mmq_mma_y_ij(
            layer.ffn_down.kind, layer.ffn_down.data, layer.ffn_down.rows,
            layer.ffn_down.columns, workspace->prompt_q8_, token_count,
            workspace->prompt_mixer_output_, selected_ffn_down_quality_i(),
            selected_ffn_down_prompt_tile(), stream,
            reinterpret_cast<float*>(workspace->prompt_projected_bf16_),
            workspace->prompt_chunk_rows_ * internal::kFfnWidth *
                sizeof(__nv_bfloat16) / sizeof(float));
      }
      if (error == cudaSuccess) error = end_phase(leaves);
    } else {
      error = begin_phase(
          leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->swiglu,
          stream);
      if (error == cudaSuccess) {
        error = launch_swiglu_bf16(
            workspace->prompt_projection_a_, workspace->prompt_projection_b_,
            token_count * internal::kFfnWidth, workspace->prompt_projected_bf16_,
            stream);
      }
      if (error == cudaSuccess) error = end_phase(leaves);
      if (error == cudaSuccess) {
        error = begin_phase(leaves,
                            leaf_timings == nullptr
                                ? nullptr
                                : &leaf_timings->proj_ffn_down,
                            stream);
      }
      if (error == cudaSuccess) {
        error = matrix_prompt(layer.ffn_down, workspace->prompt_projected_bf16_,
                              token_count, workspace,
                              workspace->prompt_mixer_output_, stream);
      }
      if (error == cudaSuccess) error = end_phase(leaves);
    }
  }
  return error;
}

cudaError_t execute_ffn(const DeviceCommonLayer& layer,
                        const float* residual,
                        SchedulerWorkspace* workspace,
                        float* output, const float* next_input_norm,
                        PointwisePath pointwise_path,
                        cudaStream_t stream,
                        GpuPhaseRecorder* leaves = nullptr,
                        LeafTimings* leaf_timings = nullptr,
                        ActivationCapture* capture = nullptr,
                        std::size_t capture_layer = 0,
                        const char* capture_kind = "") noexcept {
  if (leaf_timings != nullptr) leaf_timings->ffn_staging_fused_into_mmv = true;
  cudaError_t error = begin_phase(
      leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->ffn_norm,
      stream);
  if (error == cudaSuccess) {
    if (rms_norm_uses_serial()) {
      rms_norm_fp32_to_bf16<<<1, kThreads, 0, stream>>>(
          residual, layer.ffn_norm, internal::kResidualWidth,
          workspace->normalized_);
      error = cudaPeekAtLastError();
    } else {
      error = launch_rms_norm_fp32_dispatch(
          residual, layer.ffn_norm, internal::kResidualWidth,
          workspace->normalized_, stream);
    }
  }
  if (error == cudaSuccess) error = end_phase(leaves);
  if (error == cudaSuccess && capture != nullptr) {
    error = maybe_capture_preprojection(capture, capture_layer, false,
                                        capture_kind, workspace->normalized_,
                                        internal::kResidualWidth);
  }
  const bool paired = ffn_decode_uses_paired();
  const bool share_stage = ffn_decode_shares_stage();
  if (error == cudaSuccess) {
    error = begin_phase(
        leaves,
        leaf_timings == nullptr ? nullptr : &leaf_timings->proj_ffn_gate,
        stream);
  }
  if (error == cudaSuccess && paired) {
    if (share_stage) {
      error = launch_quantize_bf16_q8(workspace->normalized_, workspace->q8_,
                                      internal::kResidualWidth, stream);
      workspace->q8_decode_staged_activation_ = nullptr;
      workspace->q8_decode_staged_columns_ = 0;
      if (error == cudaSuccess) {
        error = launch_q4k_gate_up_swiglu_prequant(
            layer.ffn_gate.data, layer.ffn_up.data, layer.ffn_gate.rows,
            layer.ffn_gate.columns, workspace->q8_, workspace->ffn_activated_,
            stream);
      }
    } else {
      error = launch_q4k_gate_up_swiglu(
          layer.ffn_gate.data, layer.ffn_up.data, layer.ffn_gate.rows,
          layer.ffn_gate.columns, workspace->normalized_, workspace->q8_,
          workspace->ffn_activated_, stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
  } else if (error == cudaSuccess && share_stage) {
    error = launch_quantize_bf16_q8(workspace->normalized_, workspace->q8_,
                                    internal::kResidualWidth, stream);
    workspace->q8_decode_staged_activation_ = nullptr;
    workspace->q8_decode_staged_columns_ = 0;
    if (error == cudaSuccess) {
      error = launch_quant_mmv_prequant(
          layer.ffn_gate.kind, layer.ffn_gate.data, layer.ffn_gate.rows,
          layer.ffn_gate.columns, workspace->q8_, workspace->projection_a_,
          stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) {
      error = begin_phase(
          leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->proj_ffn_up,
          stream);
    }
    if (error == cudaSuccess) {
      error = launch_quant_mmv_prequant(
          layer.ffn_up.kind, layer.ffn_up.data, layer.ffn_up.rows,
          layer.ffn_up.columns, workspace->q8_, workspace->projection_b_,
          stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) {
      error = begin_phase(
          leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->swiglu,
          stream);
    }
    if (error == cudaSuccess) {
      error = launch_swiglu_bf16(
          workspace->projection_a_, workspace->projection_b_,
          internal::kFfnWidth, workspace->ffn_activated_, stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
  } else {
    if (error == cudaSuccess) {
      error = matrix_vector(layer.ffn_gate, workspace->normalized_, workspace,
                            workspace->projection_a_, stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) {
      error = begin_phase(
          leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->proj_ffn_up,
          stream);
    }
    if (error == cudaSuccess) {
      error = matrix_vector(layer.ffn_up, workspace->normalized_, workspace,
                            workspace->projection_b_, stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) {
      error = begin_phase(
          leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->swiglu,
          stream);
    }
    if (error == cudaSuccess) {
      error = launch_swiglu_bf16(
          workspace->projection_a_, workspace->projection_b_,
          internal::kFfnWidth, workspace->ffn_activated_, stream);
    }
    if (error == cudaSuccess) error = end_phase(leaves);
  }
  if (error == cudaSuccess) {
    error = begin_phase(
        leaves,
        leaf_timings == nullptr ? nullptr : &leaf_timings->proj_ffn_down,
        stream);
  }
  if (error == cudaSuccess) {
    error = matrix_vector(layer.ffn_down, workspace->ffn_activated_, workspace,
                          workspace->mixer_output_, stream);
  }
  if (error == cudaSuccess) error = end_phase(leaves);
  if (error == cudaSuccess) {
    error = begin_phase(
        leaves,
        leaf_timings == nullptr ? nullptr : &leaf_timings->residual_ffn,
        stream);
  }
  if (error == cudaSuccess) {
    if (pointwise_path == PointwisePath::kFused &&
        next_input_norm != nullptr) {
      if (rms_norm_uses_serial()) {
        residual_add_norm_fp32_to_bf16<<<1, kThreads, 0, stream>>>(
            residual, workspace->mixer_output_, next_input_norm,
            internal::kResidualWidth, output, workspace->normalized_);
        error = cudaPeekAtLastError();
      } else {
        error = launch_residual_add_norm_fp32_dispatch(
            residual, workspace->mixer_output_, next_input_norm,
            internal::kResidualWidth, output, workspace->normalized_, stream);
      }
    } else {
      residual_add_fp32<<<20, kThreads, 0, stream>>>(
          residual, workspace->mixer_output_, internal::kResidualWidth, output);
      error = cudaPeekAtLastError();
    }
  }
  if (error == cudaSuccess) error = end_phase(leaves);
  return error;
}

void bump(PromptPipelineCounters* counters,
          std::uint32_t PromptPipelineCounters::* field) noexcept {
  if (counters != nullptr) ++(counters->*field);
}

cudaError_t execute_prompt_ffn_attributed(
    const DeviceCommonLayer& layer, const float* residual,
    SchedulerWorkspace* workspace, float* after_mixer, float* output,
    const float* next_input_norm, std::size_t token_count, cudaStream_t stream,
    GpuPhaseRecorder* leaves, LeafTimings* leaf_timings,
    ActivationCapture* capture, std::size_t capture_layer,
    const char* capture_kind) noexcept {
  cudaError_t error = begin_phase(
      leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->residual_ffn,
      stream);
  if (error == cudaSuccess) {
    error = launch_residual_add_norm_rows_fp32_to_bf16(
        residual, workspace->prompt_mixer_output_, layer.ffn_norm,
        internal::kResidualWidth, token_count, after_mixer,
        workspace->prompt_normalized_, stream);
  }
  if (error == cudaSuccess) error = end_phase(leaves);
  if (error == cudaSuccess && capture != nullptr) {
    const __nv_bfloat16* row =
        workspace->prompt_normalized_ +
        (token_count - 1) * internal::kResidualWidth;
    error = maybe_capture_preprojection(capture, capture_layer, false,
                                        capture_kind, row,
                                        internal::kResidualWidth);
  }
  if (error == cudaSuccess) {
    error = execute_prompt_ffn_projections(layer, workspace, token_count, stream,
                                           leaves, leaf_timings);
  }
  if (error == cudaSuccess) {
    error = begin_phase(
        leaves, leaf_timings == nullptr ? nullptr : &leaf_timings->residual_ffn,
        stream);
  }
  if (error == cudaSuccess) {
    if (next_input_norm != nullptr) {
      error = launch_residual_add_norm_rows_fp32_to_bf16(
          after_mixer, workspace->prompt_mixer_output_, next_input_norm,
          internal::kResidualWidth, token_count, output,
          workspace->prompt_normalized_, stream);
    } else {
      error = launch_residual_add_fp32(
          after_mixer, workspace->prompt_mixer_output_,
          token_count * internal::kResidualWidth, output, stream);
    }
  }
  if (error == cudaSuccess) error = end_phase(leaves);
  return error;
}

}  // namespace

cudaError_t execute_prompt_ffn(
    const DeviceCommonLayer& layer, const float* residual,
    SchedulerWorkspace* workspace, float* after_mixer, float* output,
    const float* next_input_norm, std::size_t token_count,
    cudaStream_t stream) noexcept {
  cudaError_t error = launch_residual_add_norm_rows_fp32_to_bf16(
      residual, workspace->prompt_mixer_output_, layer.ffn_norm,
      internal::kResidualWidth, token_count, after_mixer,
      workspace->prompt_normalized_, stream);
  if (error == cudaSuccess) {
    error = execute_prompt_ffn_projections(layer, workspace, token_count,
                                           stream);
  }
  if (error == cudaSuccess) {
    if (next_input_norm != nullptr) {
      error = launch_residual_add_norm_rows_fp32_to_bf16(
          after_mixer, workspace->prompt_mixer_output_, next_input_norm,
          internal::kResidualWidth, token_count, output,
          workspace->prompt_normalized_, stream);
    } else {
      error = launch_residual_add_fp32(
          after_mixer, workspace->prompt_mixer_output_,
          token_count * internal::kResidualWidth, output, stream);
    }
  }
  return error;
}

cudaError_t launch_prepare_gdn_gate_rows(
    const float* alpha, const float* beta, const float* folded_a,
    const float* dt_bias, std::size_t key_heads, std::size_t replicas,
    std::size_t token_count, float* log_decay, float* update,
    cudaStream_t stream) noexcept {
  if (alpha == nullptr || beta == nullptr || folded_a == nullptr ||
      dt_bias == nullptr || log_decay == nullptr || update == nullptr ||
      key_heads == 0 || replicas == 0 || token_count == 0) {
    return cudaErrorInvalidValue;
  }
  return quartz_launch_kernel(
      prepare_gdn_gate_rows, dim3(static_cast<unsigned int>(token_count)),
      dim3(kThreads), 0, stream, alpha, beta, folded_a, dt_bias, key_heads,
      replicas, log_decay, update);
}

cudaError_t launch_split_attention_rows_prompt(
    const float* packed, std::size_t query_values, std::size_t head_width,
    std::size_t token_count, float* query, float* gate,
    cudaStream_t stream) noexcept {
  if (packed == nullptr || query == nullptr || gate == nullptr ||
      query_values == 0 || head_width == 0 || token_count == 0) {
    return cudaErrorInvalidValue;
  }
  const std::size_t values = token_count * query_values;
  return quartz_launch_kernel(
      split_attention_rows,
      dim3(static_cast<unsigned int>((values + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream, packed, query_values, head_width, token_count,
      query, gate);
}

cudaError_t launch_residual_add_fp32(const float* residual,
                                     const float* correction, std::size_t count,
                                     float* output,
                                     cudaStream_t stream) noexcept {
  if (residual == nullptr || correction == nullptr || output == nullptr ||
      count == 0) {
    return cudaErrorInvalidValue;
  }
  return quartz_launch_kernel(
      residual_add_fp32,
      dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream, residual, correction, count, output);
}

cudaError_t launch_rms_norm_rows_fp32_to_bf16(const float* input,
                                              const float* scale,
                                              std::size_t width,
                                              std::size_t token_count,
                                              __nv_bfloat16* output,
                                              cudaStream_t stream) noexcept {
  if (input == nullptr || scale == nullptr || output == nullptr || width == 0 ||
      token_count == 0) {
    return cudaErrorInvalidValue;
  }
  return launch_rms_norm_rows_dispatch(input, scale, width, token_count, output,
                                       stream);
}

cudaError_t launch_residual_add_norm_rows_fp32_to_bf16(
    const float* residual, const float* correction, const float* scale,
    std::size_t width, std::size_t token_count, float* output,
    __nv_bfloat16* normalized, cudaStream_t stream) noexcept {
  if (residual == nullptr || correction == nullptr || scale == nullptr ||
      output == nullptr || normalized == nullptr || width == 0 ||
      token_count == 0) {
    return cudaErrorInvalidValue;
  }
  return launch_residual_add_norm_rows_dispatch(
      residual, correction, scale, width, token_count, output, normalized,
      stream);
}

cudaError_t launch_rms_norm_fp32_to_bf16(const float* input, const float* scale,
                                         std::size_t count,
                                         __nv_bfloat16* output,
                                         cudaStream_t stream) noexcept {
  if (input == nullptr || scale == nullptr || output == nullptr || count == 0) {
    return cudaErrorInvalidValue;
  }
  return launch_rms_norm_fp32_dispatch(input, scale, count, output, stream);
}

cudaError_t launch_residual_add_norm_fp32_to_bf16(
    const float* residual, const float* correction, const float* scale,
    std::size_t count, float* output, __nv_bfloat16* normalized,
    cudaStream_t stream) noexcept {
  if (residual == nullptr || correction == nullptr || scale == nullptr ||
      output == nullptr || normalized == nullptr || count == 0) {
    return cudaErrorInvalidValue;
  }
  return launch_residual_add_norm_fp32_dispatch(
      residual, correction, scale, count, output, normalized, stream);
}

int rms_norm_parallel_occupancy(int threads) noexcept {
  int occupancy = 0;
  if (threads == 128) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, rms_norm_rows_fp32_to_bf16_parallel<false>, 128, 0);
  } else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, rms_norm_rows_fp32_to_bf16_parallel<false>, 256, 0);
  }
  return occupancy;
}

cudaError_t rms_norm_kernel_attributes(const char* path, int threads,
                                       cudaFuncAttributes* attributes) noexcept {
  if (attributes == nullptr || !legal_rms_norm_path(path) ||
      !legal_rms_norm_threads(threads)) {
    return cudaErrorInvalidValue;
  }
  if (rms_norm_path_is_serial(path)) {
    return cudaFuncGetAttributes(attributes, rms_norm_rows_fp32_to_bf16);
  }
  if (rms_norm_path_is_rsqrt(path)) {
    return cudaFuncGetAttributes(attributes,
                                 rms_norm_rows_fp32_to_bf16_parallel<true>);
  }
  return cudaFuncGetAttributes(attributes,
                               rms_norm_rows_fp32_to_bf16_parallel<false>);
}

ResidentModel::ResidentModel() noexcept = default;
ResidentModel::~ResidentModel() { release(); }

ResidentModel::ResidentModel(ResidentModel&& other) noexcept {
  *this = std::move(other);
}

ResidentModel& ResidentModel::operator=(ResidentModel&& other) noexcept {
  if (this == &other) return *this;
  release();
  blob_ = other.blob_;
  blob_bytes_ = other.blob_bytes_;
  upload_ms_ = other.upload_ms_;
  embedding_ = other.embedding_;
  output_norm_ = other.output_norm_;
  output_ = other.output_;
  layers_ = other.layers_;
  other.blob_ = nullptr;
  other.blob_bytes_ = 0;
  other.upload_ms_ = 0.0F;
  return *this;
}

void ResidentModel::release() noexcept {
  if (blob_ != nullptr) cudaFree(blob_);
  blob_ = nullptr;
  blob_bytes_ = 0;
  upload_ms_ = 0.0F;
}

Status ResidentModel::upload(const internal::ModelWeights& weights,
                             const std::uint8_t* mapped_base,
                             std::size_t mapped_bytes) noexcept {
  const NvtxRange range("qw38.loading");
  if (blob_ != nullptr || mapped_base == nullptr || mapped_bytes == 0 ||
      weights.bound_tensor_count != 851) {
    return {StatusCode::kInvalidArgument,
            "resident CUDA model upload input is invalid"};
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaMalloc(&blob_, mapped_bytes);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  if (error == cudaSuccess) {
    error = cudaMemcpy(blob_, mapped_base, mapped_bytes, cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) {
    error = cudaEventElapsedTime(&upload_ms_, start, stop);
  }
  if (stop != nullptr) cudaEventDestroy(stop);
  if (start != nullptr) cudaEventDestroy(start);
  if (error != cudaSuccess) {
    release();
    return cuda_status(error, "cannot upload the admitted GGUF to CUDA");
  }
  blob_bytes_ = mapped_bytes;
  bool valid = remap_tensor(weights.token_embedding, mapped_base, mapped_bytes,
                            blob_, &embedding_) &&
               remap_vector(weights.output_norm, mapped_base, mapped_bytes,
                            blob_, &output_norm_) &&
               remap_tensor(weights.output, mapped_base, mapped_bytes, blob_,
                            &output_);
  for (std::size_t index = 0; valid && index < layers_.size(); ++index) {
    const internal::LayerWeights& source = weights.layers[index];
    DeviceLayer& destination = layers_[index];
    destination.kind = source.kind;
    valid = remap_common(source.common, mapped_base, mapped_bytes, blob_,
                         &destination.common);
    if (!valid) break;
    if (source.kind == internal::LayerKind::kGdn) {
      valid = remap_tensor(source.gdn.packed_qkv, mapped_base, mapped_bytes,
                           blob_, &destination.gdn.packed_qkv) &&
              remap_tensor(source.gdn.value_gate, mapped_base, mapped_bytes,
                           blob_, &destination.gdn.value_gate) &&
              remap_tensor(source.gdn.alpha, mapped_base, mapped_bytes, blob_,
                           &destination.gdn.alpha) &&
              remap_tensor(source.gdn.beta, mapped_base, mapped_bytes, blob_,
                           &destination.gdn.beta) &&
              remap_vector(source.gdn.folded_a, mapped_base, mapped_bytes,
                           blob_, &destination.gdn.folded_a) &&
              remap_vector(source.gdn.dt_bias, mapped_base, mapped_bytes,
                           blob_, &destination.gdn.dt_bias) &&
              remap_vector(source.gdn.norm, mapped_base, mapped_bytes, blob_,
                           &destination.gdn.norm);
      const std::uint8_t* convolution = nullptr;
      valid = valid &&
              remap_bytes(source.gdn.convolution.data,
                          source.gdn.convolution.storage_bytes, mapped_base,
                          mapped_bytes, blob_, &convolution) &&
              remap_tensor(source.gdn.output, mapped_base, mapped_bytes, blob_,
                           &destination.gdn.output);
      destination.gdn.convolution =
          reinterpret_cast<const float*>(convolution);
    } else {
      valid = remap_tensor(source.attention.query_gate, mapped_base,
                           mapped_bytes, blob_,
                           &destination.attention.query_gate) &&
              remap_tensor(source.attention.key, mapped_base, mapped_bytes,
                           blob_, &destination.attention.key) &&
              remap_tensor(source.attention.value, mapped_base, mapped_bytes,
                           blob_, &destination.attention.value) &&
              remap_vector(source.attention.query_norm, mapped_base,
                           mapped_bytes, blob_,
                           &destination.attention.query_norm) &&
              remap_vector(source.attention.key_norm, mapped_base, mapped_bytes,
                           blob_, &destination.attention.key_norm) &&
              remap_tensor(source.attention.output, mapped_base, mapped_bytes,
                           blob_, &destination.attention.output);
    }
  }
  if (!valid) {
    release();
    return {StatusCode::kInvalidArgument,
            "typed model view does not fit the uploaded GGUF mapping"};
  }
  return Status::ok();
}

std::size_t ResidentModel::resident_bytes() const noexcept {
  return blob_bytes_;
}

float ResidentModel::upload_milliseconds() const noexcept { return upload_ms_; }

SchedulerGraphs::SchedulerGraphs() noexcept = default;
SchedulerGraphs::~SchedulerGraphs() { release(); }

SchedulerGraphs::SchedulerGraphs(SchedulerGraphs&& other) noexcept {
  *this = std::move(other);
}

SchedulerGraphs& SchedulerGraphs::operator=(SchedulerGraphs&& other) noexcept {
  if (this == &other) return *this;
  release();
  graphs_ = other.graphs_;
  executions_ = other.executions_;
  prompt_graphs_ = other.prompt_graphs_;
  prompt_executions_ = other.prompt_executions_;
  model_ = other.model_;
  workspace_ = other.workspace_;
  decode_graph_count_ = other.decode_graph_count_;
  prompt_graph_count_ = other.prompt_graph_count_;
  prompt_rows_ = other.prompt_rows_;
  allocated_bytes_ = other.allocated_bytes_;
  other.graphs_ = {};
  other.executions_ = {};
  other.prompt_graphs_ = {};
  other.prompt_executions_ = {};
  other.model_ = nullptr;
  other.workspace_ = nullptr;
  other.decode_graph_count_ = 0;
  other.prompt_graph_count_ = 0;
  other.prompt_rows_ = 0;
  other.allocated_bytes_ = 0;
  return *this;
}

void SchedulerGraphs::release() noexcept {
  for (std::size_t index = 0; index < executions_.size(); ++index) {
    if (executions_[index] != nullptr) cudaGraphExecDestroy(executions_[index]);
    if (graphs_[index] != nullptr) cudaGraphDestroy(graphs_[index]);
    executions_[index] = nullptr;
    graphs_[index] = nullptr;
  }
  for (std::size_t index = 0; index < prompt_executions_.size(); ++index) {
    if (prompt_executions_[index] != nullptr) {
      cudaGraphExecDestroy(prompt_executions_[index]);
    }
    if (prompt_graphs_[index] != nullptr) cudaGraphDestroy(prompt_graphs_[index]);
    prompt_executions_[index] = nullptr;
    prompt_graphs_[index] = nullptr;
  }
  model_ = nullptr;
  workspace_ = nullptr;
  decode_graph_count_ = 0;
  prompt_graph_count_ = 0;
  prompt_rows_ = 0;
  allocated_bytes_ = 0;
}

Status SchedulerGraphs::create(const ResidentModel& model,
                               SchedulerWorkspace* workspace) noexcept {
  const NvtxRange range("qw38.graph_create");
  if (decode_graph_count_ != 0 || prompt_graph_count_ != 0 ||
      model.blob_ == nullptr || workspace == nullptr ||
      workspace->capacity_ == 0) {
    return {StatusCode::kInvalidArgument,
            "CUDA scheduler graph creation input is invalid"};
  }
  std::size_t free_before = 0;
  std::size_t total = 0;
  cudaError_t error = cudaMemGetInfo(&free_before, &total);
  cudaStream_t stream = nullptr;
  if (error == cudaSuccess) {
    error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  for (std::size_t layer_index = 0;
       error == cudaSuccess && layer_index < model.layers_.size();
       ++layer_index) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    cudaError_t enqueue_error = error;
    if (enqueue_error == cudaSuccess) {
      const float* next_input_norm =
          layer_index + 1 < model.layers_.size()
              ? model.layers_[layer_index + 1].common.input_norm
              : nullptr;
      enqueue_error = execute_ffn(
          model.layers_[layer_index].common, workspace->residual_b_, workspace,
          workspace->residual_a_, next_input_norm, PointwisePath::kFused,
          stream);
    }
    cudaError_t capture_error = cudaStreamEndCapture(
        stream, &graphs_[layer_index]);
    error = enqueue_error != cudaSuccess ? enqueue_error : capture_error;
    if (error == cudaSuccess) {
      error = cudaGraphInstantiate(&executions_[layer_index],
                                   graphs_[layer_index], nullptr, nullptr, 0);
    }
    if (error == cudaSuccess) {
      error = cudaGraphUpload(executions_[layer_index], stream);
    }
    if (error == cudaSuccess) ++decode_graph_count_;
  }
  if (error == cudaSuccess &&
      workspace->prompt_chunk_rows_ == kPromptChunkRows) {
    for (std::size_t layer_index = 0;
         error == cudaSuccess && layer_index < model.layers_.size();
         ++layer_index) {
      error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
      cudaError_t enqueue_error = error;
      if (enqueue_error == cudaSuccess) {
        const float* next_input_norm =
            layer_index + 1 < model.layers_.size()
                ? model.layers_[layer_index + 1].common.input_norm
                : nullptr;
        enqueue_error = execute_prompt_ffn(
            model.layers_[layer_index].common, workspace->prompt_residual_a_,
            workspace, workspace->prompt_residual_b_,
            workspace->prompt_residual_a_, next_input_norm, kPromptChunkRows,
            stream);
      }
      cudaError_t capture_error =
          cudaStreamEndCapture(stream, &prompt_graphs_[layer_index]);
      error = enqueue_error != cudaSuccess ? enqueue_error : capture_error;
      if (error == cudaSuccess) {
        error = cudaGraphInstantiate(&prompt_executions_[layer_index],
                                     prompt_graphs_[layer_index], nullptr,
                                     nullptr, 0);
      }
      if (error == cudaSuccess) {
        error = cudaGraphUpload(prompt_executions_[layer_index], stream);
      }
      if (error == cudaSuccess) ++prompt_graph_count_;
    }
    if (error == cudaSuccess &&
        prompt_graph_count_ == internal::kModelLayerCount) {
      prompt_rows_ = kPromptChunkRows;
    }
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  std::size_t free_after = 0;
  if (error == cudaSuccess) error = cudaMemGetInfo(&free_after, &total);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (error != cudaSuccess) {
    release();
    return cuda_status(error, "cannot capture CUDA scheduler FFN graphs");
  }
  model_ = &model;
  workspace_ = workspace;
  allocated_bytes_ = free_before >= free_after ? free_before - free_after : 0;
  return Status::ok();
}

std::size_t SchedulerGraphs::graph_count() const noexcept {
  return decode_graph_count_ + prompt_graph_count_;
}

std::size_t SchedulerGraphs::decode_graph_count() const noexcept {
  return decode_graph_count_;
}

std::size_t SchedulerGraphs::prompt_graph_count() const noexcept {
  return prompt_graph_count_;
}

std::size_t SchedulerGraphs::prompt_graph_rows() const noexcept {
  return prompt_rows_;
}

std::size_t SchedulerGraphs::allocated_bytes() const noexcept {
  return allocated_bytes_;
}

bool SchedulerGraphs::matches(
    const ResidentModel& model,
    const SchedulerWorkspace* workspace) const noexcept {
  return decode_graph_count_ == internal::kModelLayerCount &&
         model_ == &model && workspace_ == workspace;
}

SchedulerSession::SchedulerSession() noexcept = default;
SchedulerSession::~SchedulerSession() { release(); }
SchedulerSession::SchedulerSession(SchedulerSession&& other) noexcept {
  *this = std::move(other);
}

SchedulerSession& SchedulerSession::operator=(SchedulerSession&& other) noexcept {
  if (this == &other) return *this;
  release();
  gdn_convolution_ = other.gdn_convolution_;
  gdn_recurrent_ = other.gdn_recurrent_;
  attention_key_ = other.attention_key_;
  attention_value_ = other.attention_value_;
  tokens_ = other.tokens_;
  last_logits_ = other.last_logits_;
  last_hidden_ = other.last_hidden_;
  capacity_ = other.capacity_;
  frontier_ = other.frontier_;
  allocated_bytes_ = other.allocated_bytes_;
  sampler_state_ = other.sampler_state_;
  other.gdn_convolution_ = nullptr;
  other.gdn_recurrent_ = nullptr;
  other.attention_key_ = nullptr;
  other.attention_value_ = nullptr;
  other.tokens_ = nullptr;
  other.last_logits_ = nullptr;
  other.last_hidden_ = nullptr;
  other.capacity_ = 0;
  other.frontier_ = 0;
  other.allocated_bytes_ = 0;
  other.sampler_state_ = {};
  return *this;
}

void SchedulerSession::release() noexcept {
  std::free(last_hidden_);
  std::free(last_logits_);
  std::free(tokens_);
  if (attention_value_ != nullptr) cudaFree(attention_value_);
  if (attention_key_ != nullptr) cudaFree(attention_key_);
  if (gdn_recurrent_ != nullptr) cudaFree(gdn_recurrent_);
  if (gdn_convolution_ != nullptr) cudaFree(gdn_convolution_);
  gdn_convolution_ = nullptr;
  gdn_recurrent_ = nullptr;
  attention_key_ = nullptr;
  attention_value_ = nullptr;
  tokens_ = nullptr;
  last_logits_ = nullptr;
  last_hidden_ = nullptr;
  capacity_ = 0;
  frontier_ = 0;
  allocated_bytes_ = 0;
  sampler_state_ = {};
}

Status SchedulerSession::create(std::size_t capacity) noexcept {
  if (capacity == 0 || capacity > 131072 || capacity_ != 0) {
    return {StatusCode::kInvalidArgument,
            "CUDA scheduler session capacity is invalid"};
  }
  tokens_ = static_cast<std::size_t*>(
      std::malloc(capacity * sizeof(std::size_t)));
  last_logits_ = static_cast<float*>(
      std::malloc(internal::kVocabularySize * sizeof(float)));
  last_hidden_ = static_cast<float*>(
      std::malloc(internal::kResidualWidth * sizeof(float)));
  if (tokens_ == nullptr || last_logits_ == nullptr || last_hidden_ == nullptr) {
    release();
    return {StatusCode::kResourceExhausted,
            "cannot allocate CUDA scheduler host session state"};
  }
  cudaError_t error = allocate(&gdn_convolution_,
                               kGdnLayers * internal::kGdnConvolutionValues,
                               &allocated_bytes_);
  if (error == cudaSuccess) {
    error = allocate(&gdn_recurrent_,
                     kGdnLayers * internal::kGdnRecurrentStateValues,
                     &allocated_bytes_);
  }
  const std::size_t attention_values =
      kAttentionLayers * capacity * internal::kAttentionKvWidth;
  if (error == cudaSuccess) {
    error = allocate(&attention_key_, attention_values, &allocated_bytes_);
  }
  if (error == cudaSuccess) {
    error = allocate(&attention_value_, attention_values, &allocated_bytes_);
  }
  if (error == cudaSuccess) {
    error = cudaMemset(gdn_convolution_, 0,
                       kGdnLayers * internal::kGdnConvolutionValues *
                           sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(gdn_recurrent_, 0,
                       kGdnLayers * internal::kGdnRecurrentStateValues *
                           sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(attention_key_, 0,
                       attention_values * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(attention_value_, 0,
                       attention_values * sizeof(__nv_bfloat16));
  }
  if (error != cudaSuccess) {
    release();
    return cuda_status(error, "cannot allocate CUDA scheduler session state");
  }
  capacity_ = capacity;
  return Status::ok();
}

Status SchedulerSession::reset() noexcept {
  if (capacity_ == 0 || tokens_ == nullptr) {
    return {StatusCode::kInvalidArgument,
            "CUDA scheduler session is not initialized"};
  }
  const std::size_t attention_values =
      kAttentionLayers * capacity_ * internal::kAttentionKvWidth;
  cudaError_t error = cudaMemset(
      gdn_convolution_, 0,
      kGdnLayers * internal::kGdnConvolutionValues * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMemset(
        gdn_recurrent_, 0,
        kGdnLayers * internal::kGdnRecurrentStateValues * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(attention_key_, 0,
                       attention_values * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(attention_value_, 0,
                       attention_values * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    return cuda_status(error, "cannot reset CUDA scheduler session state");
  }
  frontier_ = 0;
  return Status::ok();
}

Status SchedulerSession::state_equals(const SchedulerSession& other,
                                      bool* equal) const noexcept {
  if (equal == nullptr || capacity_ == 0 || other.capacity_ == 0) {
    return {StatusCode::kInvalidArgument,
            "CUDA scheduler state comparison input is invalid"};
  }
  *equal = false;
  if (capacity_ != other.capacity_ || frontier_ != other.frontier_ ||
      sampler_state_.temperature != other.sampler_state_.temperature ||
      sampler_state_.top_p != other.sampler_state_.top_p ||
      sampler_state_.top_k != other.sampler_state_.top_k ||
      sampler_state_.seed != other.sampler_state_.seed ||
      sampler_state_.rng_state != other.sampler_state_.rng_state ||
      std::memcmp(tokens_, other.tokens_, frontier_ * sizeof(std::size_t)) != 0) {
    return Status::ok();
  }
  if (frontier_ > 0 &&
      (std::memcmp(last_logits_, other.last_logits_,
                   internal::kVocabularySize * sizeof(float)) != 0 ||
       std::memcmp(last_hidden_, other.last_hidden_,
                   internal::kResidualWidth * sizeof(float)) != 0)) {
    return Status::ok();
  }
  unsigned int* mismatch = nullptr;
  cudaError_t error = cudaMalloc(&mismatch, sizeof(unsigned int));
  if (error == cudaSuccess) error = cudaMemset(mismatch, 0, sizeof(unsigned int));
  const auto compare = [&error, mismatch](const void* left, const void* right,
                                         std::size_t bytes) {
    if (error != cudaSuccess) return;
    compare_bytes<<<256, kThreads>>>(
        static_cast<const std::uint8_t*>(left),
        static_cast<const std::uint8_t*>(right), bytes, mismatch);
    error = cudaPeekAtLastError();
  };
  compare(gdn_convolution_, other.gdn_convolution_,
          kGdnLayers * internal::kGdnConvolutionValues * sizeof(float));
  compare(gdn_recurrent_, other.gdn_recurrent_,
          kGdnLayers * internal::kGdnRecurrentStateValues * sizeof(float));
  const std::size_t cache_stride = capacity_ * internal::kAttentionKvWidth;
  const std::size_t head_span_bytes =
      frontier_ * internal::kAttentionHeadWidth * sizeof(__nv_bfloat16);
  const std::size_t kv_heads =
      internal::kAttentionKvWidth / internal::kAttentionHeadWidth;
  for (std::size_t layer = 0; layer < kAttentionLayers; ++layer) {
    for (std::size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      const std::size_t head_offset =
          kv_head * capacity_ * internal::kAttentionHeadWidth;
      compare(attention_key_ + layer * cache_stride + head_offset,
              other.attention_key_ + layer * cache_stride + head_offset,
              head_span_bytes);
      compare(attention_value_ + layer * cache_stride + head_offset,
              other.attention_value_ + layer * cache_stride + head_offset,
              head_span_bytes);
    }
  }
  unsigned int host_mismatch = 1;
  if (error == cudaSuccess) {
    error = cudaMemcpy(&host_mismatch, mismatch, sizeof(unsigned int),
                       cudaMemcpyDeviceToHost);
  }
  if (mismatch != nullptr) cudaFree(mismatch);
  if (error != cudaSuccess) {
    return cuda_status(error, "cannot compare CUDA scheduler session state");
  }
  *equal = host_mismatch == 0;
  return Status::ok();
}

Status SchedulerSession::copy_last_outputs(
    float* logits, std::size_t logits_count, float* hidden,
    std::size_t hidden_count) const noexcept {
  if (frontier_ == 0 || logits == nullptr || hidden == nullptr ||
      logits_count != internal::kVocabularySize ||
      hidden_count != internal::kResidualWidth) {
    return {StatusCode::kInvalidArgument,
            "CUDA committed output copy input is invalid"};
  }
  std::memcpy(logits, last_logits_, logits_count * sizeof(float));
  std::memcpy(hidden, last_hidden_, hidden_count * sizeof(float));
  return Status::ok();
}

Status SchedulerSession::copy_tokens(std::size_t* output,
                                     std::size_t output_count) const noexcept {
  if ((frontier_ > 0 && output == nullptr) || output_count != frontier_) {
    return {StatusCode::kInvalidArgument,
            "CUDA committed token copy input is invalid"};
  }
  if (frontier_ > 0) {
    std::memcpy(output, tokens_, frontier_ * sizeof(std::size_t));
  }
  return Status::ok();
}

std::size_t SchedulerSession::capacity() const noexcept { return capacity_; }
std::size_t SchedulerSession::frontier() const noexcept { return frontier_; }
std::size_t SchedulerSession::token_count() const noexcept { return frontier_; }
std::size_t SchedulerSession::allocated_bytes() const noexcept {
  return allocated_bytes_;
}

Status SchedulerSession::set_sampler_state(const SamplerState& state) noexcept {
  if (!std::isfinite(state.temperature) || state.temperature < 0.0F ||
      !std::isfinite(state.top_p) || state.top_p <= 0.0F ||
      state.top_p > 1.0F || state.top_k > internal::kVocabularySize) {
    return {StatusCode::kInvalidArgument, "CUDA sampler state is invalid"};
  }
  sampler_state_ = state;
  return Status::ok();
}

SamplerState SchedulerSession::sampler_state() const noexcept {
  return sampler_state_;
}

SchedulerWorkspace::SchedulerWorkspace() noexcept = default;
SchedulerWorkspace::~SchedulerWorkspace() { release(); }
SchedulerWorkspace::SchedulerWorkspace(SchedulerWorkspace&& other) noexcept {
  *this = std::move(other);
}

SchedulerWorkspace& SchedulerWorkspace::operator=(
    SchedulerWorkspace&& other) noexcept {
  if (this == &other) return *this;
  release();
#define QW38_MOVE_POINTER(name) name = other.name; other.name = nullptr
  QW38_MOVE_POINTER(residual_a_);
  QW38_MOVE_POINTER(residual_b_);
  QW38_MOVE_POINTER(normalized_);
  QW38_MOVE_POINTER(projected_bf16_);
  QW38_MOVE_POINTER(ffn_activated_);
  QW38_MOVE_POINTER(q8_);
  q8_decode_staged_activation_ = other.q8_decode_staged_activation_;
  q8_decode_staged_columns_ = other.q8_decode_staged_columns_;
  other.q8_decode_staged_activation_ = nullptr;
  other.q8_decode_staged_columns_ = 0;
  QW38_MOVE_POINTER(projection_a_);
  QW38_MOVE_POINTER(projection_b_);
  QW38_MOVE_POINTER(projection_c_);
  QW38_MOVE_POINTER(projection_d_);
  QW38_MOVE_POINTER(mixer_output_);
  QW38_MOVE_POINTER(gdn_decay_);
  QW38_MOVE_POINTER(gdn_update_);
  QW38_MOVE_POINTER(gdn_convolved_);
  QW38_MOVE_POINTER(gdn_recurrent_output_);
  QW38_MOVE_POINTER(gdn_candidate_convolution_);
  QW38_MOVE_POINTER(gdn_candidate_recurrent_);
  QW38_MOVE_POINTER(attention_candidate_key_);
  QW38_MOVE_POINTER(attention_candidate_value_);
  QW38_MOVE_POINTER(attention_normalized_query_);
  QW38_MOVE_POINTER(attention_normalized_key_);
  QW38_MOVE_POINTER(attention_scores_);
  QW38_MOVE_POINTER(logits_);
#ifdef QW38_DIAGNOSTIC_TRACE
  QW38_MOVE_POINTER(trace_taps_);
#endif
  QW38_MOVE_POINTER(candidate_logits_host_);
  QW38_MOVE_POINTER(candidate_hidden_host_);
  QW38_MOVE_POINTER(prompt_residual_a_);
  QW38_MOVE_POINTER(prompt_residual_b_);
  QW38_MOVE_POINTER(prompt_normalized_);
  QW38_MOVE_POINTER(prompt_projected_bf16_);
  QW38_MOVE_POINTER(prompt_q8_);
  QW38_MOVE_POINTER(prompt_projection_a_);
  QW38_MOVE_POINTER(prompt_projection_b_);
  QW38_MOVE_POINTER(prompt_projection_c_);
  QW38_MOVE_POINTER(prompt_projection_d_);
  QW38_MOVE_POINTER(prompt_mixer_output_);
  QW38_MOVE_POINTER(prompt_gdn_decay_);
  QW38_MOVE_POINTER(prompt_gdn_update_);
  QW38_MOVE_POINTER(prompt_gdn_convolved_);
  QW38_MOVE_POINTER(prompt_gdn_recurrent_output_);
  QW38_MOVE_POINTER(prompt_attention_candidate_key_);
  QW38_MOVE_POINTER(prompt_attention_candidate_value_);
#undef QW38_MOVE_POINTER
  capacity_ = other.capacity_;
  prompt_chunk_rows_ = other.prompt_chunk_rows_;
  allocated_bytes_ = other.allocated_bytes_;
  prompt_compute_stream_ = other.prompt_compute_stream_;
  prompt_copy_stream_ = other.prompt_copy_stream_;
  prompt_compute_done_ = other.prompt_compute_done_;
  other.capacity_ = 0;
  other.prompt_chunk_rows_ = 0;
  other.allocated_bytes_ = 0;
  other.prompt_compute_stream_ = nullptr;
  other.prompt_copy_stream_ = nullptr;
  other.prompt_compute_done_ = nullptr;
  return *this;
}

void SchedulerWorkspace::release() noexcept {
  std::free(candidate_hidden_host_);
  std::free(candidate_logits_host_);
  candidate_hidden_host_ = nullptr;
  candidate_logits_host_ = nullptr;
#define QW38_FREE(name) if (name != nullptr) cudaFree(name); name = nullptr
  QW38_FREE(prompt_attention_candidate_value_);
  QW38_FREE(prompt_attention_candidate_key_);
  QW38_FREE(prompt_gdn_recurrent_output_);
  QW38_FREE(prompt_gdn_convolved_);
  QW38_FREE(prompt_gdn_update_);
  QW38_FREE(prompt_gdn_decay_);
  QW38_FREE(prompt_mixer_output_);
  QW38_FREE(prompt_projection_d_);
  QW38_FREE(prompt_projection_c_);
  QW38_FREE(prompt_projection_b_);
  QW38_FREE(prompt_projection_a_);
  QW38_FREE(prompt_q8_);
  QW38_FREE(prompt_projected_bf16_);
  QW38_FREE(prompt_normalized_);
  QW38_FREE(prompt_residual_b_);
  QW38_FREE(prompt_residual_a_);
#ifdef QW38_DIAGNOSTIC_TRACE
  QW38_FREE(trace_taps_);
#endif
  QW38_FREE(logits_);
  QW38_FREE(attention_scores_);
  QW38_FREE(attention_normalized_key_);
  QW38_FREE(attention_normalized_query_);
  QW38_FREE(attention_candidate_value_);
  QW38_FREE(attention_candidate_key_);
  QW38_FREE(gdn_candidate_recurrent_);
  QW38_FREE(gdn_candidate_convolution_);
  QW38_FREE(gdn_recurrent_output_);
  QW38_FREE(gdn_convolved_);
  QW38_FREE(gdn_update_);
  QW38_FREE(gdn_decay_);
  QW38_FREE(mixer_output_);
  QW38_FREE(projection_d_);
  QW38_FREE(projection_c_);
  QW38_FREE(projection_b_);
  QW38_FREE(projection_a_);
  QW38_FREE(q8_);
  QW38_FREE(ffn_activated_);
  QW38_FREE(projected_bf16_);
  QW38_FREE(normalized_);
  QW38_FREE(residual_b_);
  QW38_FREE(residual_a_);
#undef QW38_FREE
  if (prompt_compute_done_ != nullptr) {
    cudaEventDestroy(prompt_compute_done_);
    prompt_compute_done_ = nullptr;
  }
  if (prompt_compute_stream_ != nullptr) {
    cudaStreamDestroy(prompt_compute_stream_);
    prompt_compute_stream_ = nullptr;
  }
  if (prompt_copy_stream_ != nullptr) {
    cudaStreamDestroy(prompt_copy_stream_);
    prompt_copy_stream_ = nullptr;
  }
  capacity_ = 0;
  prompt_chunk_rows_ = 0;
  allocated_bytes_ = 0;
  q8_decode_staged_activation_ = nullptr;
  q8_decode_staged_columns_ = 0;
}

Status SchedulerWorkspace::create(std::size_t capacity) noexcept {
  if (capacity == 0 || capacity > 131072 || capacity_ != 0) {
    return {StatusCode::kInvalidArgument,
            "CUDA scheduler workspace capacity is invalid"};
  }
  candidate_logits_host_ = static_cast<float*>(
      std::malloc(internal::kVocabularySize * sizeof(float)));
  candidate_hidden_host_ = static_cast<float*>(
      std::malloc(internal::kResidualWidth * sizeof(float)));
  if (candidate_logits_host_ == nullptr || candidate_hidden_host_ == nullptr) {
    release();
    return {StatusCode::kResourceExhausted,
            "cannot allocate CUDA scheduler host candidate output"};
  }
  prompt_chunk_rows_ = std::min(kPromptChunkRows, capacity);
  cudaError_t error = allocate(&residual_a_, internal::kResidualWidth,
                               &allocated_bytes_);
#define QW38_ALLOCATE(name, count)                                           \
  if (error == cudaSuccess) error = allocate(&(name), (count), &allocated_bytes_)
  QW38_ALLOCATE(residual_b_, internal::kResidualWidth);
  QW38_ALLOCATE(normalized_, internal::kResidualWidth);
  QW38_ALLOCATE(projected_bf16_, internal::kGdnValueWidth);
  QW38_ALLOCATE(ffn_activated_, internal::kFfnWidth);
  QW38_ALLOCATE(q8_, internal::kFfnWidth / 32);
  QW38_ALLOCATE(projection_a_, kMaximumProjection);
  QW38_ALLOCATE(projection_b_, kMaximumProjection);
  QW38_ALLOCATE(projection_c_, internal::kAttentionKvWidth);
  QW38_ALLOCATE(projection_d_, internal::kAttentionKvWidth);
  QW38_ALLOCATE(mixer_output_, internal::kResidualWidth);
  QW38_ALLOCATE(gdn_decay_, internal::kGdnGateCount);
  QW38_ALLOCATE(gdn_update_, internal::kGdnGateCount);
  QW38_ALLOCATE(gdn_convolved_, internal::kGdnPackedQkvWidth);
  QW38_ALLOCATE(gdn_recurrent_output_, internal::kGdnValueWidth);
  QW38_ALLOCATE(gdn_candidate_convolution_,
                kGdnLayers * internal::kGdnConvolutionValues);
  QW38_ALLOCATE(gdn_candidate_recurrent_,
                kGdnLayers * internal::kGdnRecurrentStateValues);
  QW38_ALLOCATE(attention_candidate_key_,
                kAttentionLayers * internal::kAttentionKvWidth);
  QW38_ALLOCATE(attention_candidate_value_,
                kAttentionLayers * internal::kAttentionKvWidth);
  QW38_ALLOCATE(attention_normalized_query_, internal::kAttentionQueryWidth);
  QW38_ALLOCATE(attention_normalized_key_, internal::kAttentionKvWidth);
  QW38_ALLOCATE(attention_scores_, 24 * capacity);
  QW38_ALLOCATE(logits_, internal::kVocabularySize);
#ifdef QW38_DIAGNOSTIC_TRACE
  QW38_ALLOCATE(trace_taps_, kTraceTapCount * internal::kResidualWidth);
#endif
  QW38_ALLOCATE(prompt_residual_a_,
                prompt_chunk_rows_ * internal::kResidualWidth);
  QW38_ALLOCATE(prompt_residual_b_,
                prompt_chunk_rows_ * internal::kResidualWidth);
  QW38_ALLOCATE(prompt_normalized_,
                prompt_chunk_rows_ * internal::kResidualWidth);
  QW38_ALLOCATE(prompt_projected_bf16_,
                prompt_chunk_rows_ * internal::kFfnWidth);
  QW38_ALLOCATE(prompt_q8_,
                prompt_chunk_rows_ * internal::kFfnWidth / 32);
  QW38_ALLOCATE(prompt_projection_a_,
                prompt_chunk_rows_ * kMaximumProjection);
  QW38_ALLOCATE(prompt_projection_b_,
                prompt_chunk_rows_ * kMaximumProjection);
  QW38_ALLOCATE(prompt_projection_c_,
                prompt_chunk_rows_ * internal::kAttentionKvWidth);
  QW38_ALLOCATE(prompt_projection_d_,
                prompt_chunk_rows_ * internal::kAttentionKvWidth);
  QW38_ALLOCATE(prompt_mixer_output_,
                prompt_chunk_rows_ * internal::kResidualWidth);
  QW38_ALLOCATE(prompt_gdn_decay_,
                prompt_chunk_rows_ * internal::kGdnGateCount);
  QW38_ALLOCATE(prompt_gdn_update_,
                prompt_chunk_rows_ * internal::kGdnGateCount);
  QW38_ALLOCATE(prompt_gdn_convolved_,
                prompt_chunk_rows_ * internal::kGdnPackedQkvWidth);
  QW38_ALLOCATE(prompt_gdn_recurrent_output_,
                prompt_chunk_rows_ * internal::kGdnValueWidth);
  QW38_ALLOCATE(prompt_attention_candidate_key_,
                kAttentionLayers * prompt_chunk_rows_ *
                    internal::kAttentionKvWidth);
  QW38_ALLOCATE(prompt_attention_candidate_value_,
                kAttentionLayers * prompt_chunk_rows_ *
                    internal::kAttentionKvWidth);
#undef QW38_ALLOCATE
  if (error != cudaSuccess) {
    release();
    return cuda_status(error, "cannot allocate CUDA scheduler workspace");
  }
  error = cudaStreamCreateWithFlags(&prompt_compute_stream_,
                                    cudaStreamNonBlocking);
  if (error == cudaSuccess) {
    error = cudaStreamCreateWithFlags(&prompt_copy_stream_,
                                      cudaStreamNonBlocking);
  }
  if (error == cudaSuccess) {
    error = cudaEventCreateWithFlags(&prompt_compute_done_,
                                     cudaEventDisableTiming);
  }
  if (error != cudaSuccess) {
    release();
    return cuda_status(error, "cannot allocate CUDA scheduler workspace");
  }
  capacity_ = capacity;
  return Status::ok();
}

std::size_t SchedulerWorkspace::allocated_bytes() const noexcept {
  return allocated_bytes_;
}

#ifdef QW38_DIAGNOSTIC_TRACE
Status SchedulerWorkspace::copy_trace_taps(float* output,
                                           std::size_t count) const noexcept {
  if (trace_taps_ == nullptr || output == nullptr ||
      count != kTraceTapCount * internal::kResidualWidth) {
    return {StatusCode::kInvalidArgument,
            "CUDA scheduler trace-tap output is invalid"};
  }
  return cuda_status(
      cudaMemcpy(output, trace_taps_, count * sizeof(float),
                 cudaMemcpyDeviceToHost),
      "cannot copy CUDA scheduler trace taps");
}
#endif

Status execute_token(const ResidentModel& model, std::size_t token,
                     SchedulerSession* session, SchedulerWorkspace* workspace,
                     float* host_logits, std::size_t logits_count,
                     float* host_hidden, std::size_t hidden_count,
                     float* elapsed_milliseconds,
                     const EvalControl* control,
                     RuntimeTimings* timings,
                     PointwisePath pointwise_path,
                     SchedulerGraphs* graphs,
                     DecodeAttribution* decode_attribution) noexcept {
  if (model.blob_ == nullptr || token >= internal::kVocabularySize ||
      session == nullptr || workspace == nullptr ||
      session->capacity_ == 0 || session->frontier_ >= session->capacity_ ||
      workspace->capacity_ != session->capacity_ || host_logits == nullptr ||
      logits_count != internal::kVocabularySize || host_hidden == nullptr ||
      hidden_count != internal::kResidualWidth ||
      elapsed_milliseconds == nullptr ||
      (pointwise_path != PointwisePath::kFused &&
       pointwise_path != PointwisePath::kUnfused) ||
      (graphs != nullptr &&
       (pointwise_path != PointwisePath::kFused ||
        !graphs->matches(model, workspace)))) {
    return {StatusCode::kInvalidArgument,
            "CUDA token scheduler input, state, or output is invalid"};
  }
  const NvtxRange token_range("qw38.token");
  const auto attribution_started = std::chrono::steady_clock::now();
  const bool record_leaves =
      decode_attribution != nullptr && decode_attribution->record_leaves;
  ActivationCapture* capture =
      decode_attribution == nullptr ? nullptr : decode_attribution->capture;
  if (decode_attribution != nullptr) {
    *decode_attribution = {};
    decode_attribution->record_leaves = record_leaves;
    decode_attribution->capture = capture;
  }
  GpuPhaseRecorder category_recorder;
  GpuPhaseRecorder leaf_recorder;
  GpuPhaseRecorder total_recorder;
  const bool record_categories =
      timings != nullptr || decode_attribution != nullptr;
  GpuPhaseRecorder* categories =
      record_categories ? &category_recorder : nullptr;
  GpuPhaseRecorder* leaves =
      record_leaves ? &leaf_recorder : nullptr;
  LeafTimings* leaf_timings =
      record_leaves ? &decode_attribution->leaves : nullptr;
  GpuPhaseRecorder* total = timings == nullptr ? nullptr : &total_recorder;
  if (timings != nullptr) {
    *timings = {};
    timings->loading = {model.upload_milliseconds(), true};
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  if (error == cudaSuccess) {
    error = begin_phase(total, timings == nullptr ? nullptr
                                                  : &timings->token_total);
  }
  if (error == cudaSuccess) {
    error = begin_phase(categories,
                        decode_attribution == nullptr
                            ? (timings == nullptr ? nullptr : &timings->embedding)
                            : &decode_attribution->embedding);
  }
  nvtxRangePushA("qw38.embedding");
  if (error == cudaSuccess) {
    error = begin_phase(leaves, leaf_timings == nullptr
                                    ? nullptr
                                    : &leaf_timings->embedding);
  }
  if (error == cudaSuccess) {
    error = launch_quant_row_decode(
        model.embedding_.kind, model.embedding_.data, model.embedding_.rows,
        model.embedding_.columns, token, workspace->normalized_, nullptr);
  }
  if (error == cudaSuccess) {
    bf16_to_fp32<<<20, kThreads>>>(workspace->normalized_,
                                   internal::kResidualWidth,
                                   workspace->residual_a_);
    error = cudaPeekAtLastError();
  }
  if (error == cudaSuccess) error = end_phase(leaves);
  if (error == cudaSuccess) error = end_phase(categories);
  nvtxRangePop();
  if (capture != nullptr) capture->position = session->frontier_;
  std::size_t gdn_slot = 0;
  std::size_t attention_slot = 0;
  Status poll_status = Status::ok();
  bool interrupted = false;
  float* residual = workspace->residual_a_;
  float* next = workspace->residual_b_;
  for (std::size_t layer_index = 0;
       error == cudaSuccess && !interrupted &&
       layer_index < model.layers_.size();
       ++layer_index) {
    const DeviceLayer& layer = model.layers_[layer_index];
    const bool gdn_layer = layer.kind == internal::LayerKind::kGdn;
    const char* layer_kind = gdn_layer ? "gdn" : "attention";
    const bool exclusive = decode_attribution != nullptr;
    TimingValue* mixer_composite =
        timings == nullptr ? nullptr
                           : (gdn_layer ? &timings->gdn : &timings->attention);
    TimingValue* mixer_mmv =
        exclusive ? &decode_attribution->mixer_mmv : nullptr;
    TimingValue* core =
        exclusive ? (gdn_layer ? &decode_attribution->gdn_core
                               : &decode_attribution->attention_core)
                  : nullptr;
    const char* core_nvtx = gdn_layer ? "qw38.gdn_core" : "qw38.attention_core";
    nvtxRangePushA(gdn_layer ? "qw38.gdn" : "qw38.attention");
    if (!exclusive && error == cudaSuccess) {
      error = begin_phase(categories, mixer_composite);
    }
    const bool launch_rms =
        layer_index == 0 || pointwise_path == PointwisePath::kUnfused;
    if (launch_rms) {
      const NvtxRange core_range(core_nvtx);
      if (exclusive && error == cudaSuccess) {
        error = begin_phase(categories, core);
      }
      if (error == cudaSuccess) {
        error = begin_phase(leaves, leaf_timings == nullptr
                                        ? nullptr
                                        : &leaf_timings->input_norm);
      }
      if (error == cudaSuccess) {
        error = launch_rms_norm_fp32_dispatch(
            residual, layer.common.input_norm, internal::kResidualWidth,
            workspace->normalized_, nullptr);
      }
      if (error == cudaSuccess) error = end_phase(leaves);
      if (exclusive && error == cudaSuccess) error = end_phase(categories);
    }
    if (error == cudaSuccess) {
      error = maybe_capture_preprojection(
          capture, layer_index, true, layer_kind, workspace->normalized_,
          internal::kResidualWidth);
    }
    {
      const NvtxRange mixer_range("qw38.mixer_mmv");
      if (exclusive && error == cudaSuccess) {
        error = begin_phase(categories, mixer_mmv);
      }
      if (gdn_layer) {
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_packed_qkv);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.gdn.packed_qkv, workspace->normalized_,
                                workspace, workspace->projection_a_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_value_gate);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.gdn.value_gate, workspace->normalized_,
                                workspace, workspace->projection_b_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_alpha);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.gdn.alpha, workspace->normalized_,
                                workspace, workspace->projection_c_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_beta);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.gdn.beta, workspace->normalized_,
                                workspace, workspace->projection_d_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
      } else {
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_query_gate);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.attention.query_gate,
                                workspace->normalized_, workspace,
                                workspace->projection_a_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_key);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.attention.key, workspace->normalized_,
                                workspace, workspace->projection_c_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_value);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.attention.value, workspace->normalized_,
                                workspace, workspace->projection_d_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
      }
      if (exclusive && error == cudaSuccess) error = end_phase(categories);
    }
    {
      const NvtxRange core_range(core_nvtx);
      if (exclusive && error == cudaSuccess) {
        error = begin_phase(categories, core);
      }
      if (gdn_layer) {
        if (leaf_timings != nullptr) {
          leaf_timings->gdn_conv_fused_with_recurrence = true;
          leaf_timings->gdn_output_norm_fused_with_gate = true;
        }
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->gdn_gate_prep);
        }
        if (error == cudaSuccess) {
          error = launch_prepare_gdn_gates(
              workspace->projection_c_, workspace->projection_d_,
              layer.gdn.folded_a, layer.gdn.dt_bias, 16, 3,
              workspace->gdn_decay_, workspace->gdn_update_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        const GdnState committed{
            session->gdn_convolution_ +
                gdn_slot * internal::kGdnConvolutionValues,
            session->gdn_recurrent_ +
                gdn_slot * internal::kGdnRecurrentStateValues};
        const GdnState candidate{
            workspace->gdn_candidate_convolution_ +
                gdn_slot * internal::kGdnConvolutionValues,
            workspace->gdn_candidate_recurrent_ +
                gdn_slot * internal::kGdnRecurrentStateValues};
        if (error == cudaSuccess) {
          error = begin_phase(
              leaves, leaf_timings == nullptr
                          ? nullptr
                          : &leaf_timings->gdn_conv_qk_norm_recurrence);
        }
        if (error == cudaSuccess) {
          error = launch_gdn_prepare_tiled(
              kGdnConfig, workspace->projection_a_, layer.gdn.convolution,
              workspace->gdn_decay_, workspace->gdn_update_, committed,
              candidate, workspace->gdn_convolved_,
              workspace->gdn_recurrent_output_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->gdn_output_norm);
        }
        if (error == cudaSuccess) {
          error = launch_gdn_gated_output(
              workspace->gdn_recurrent_output_, workspace->projection_b_,
              layer.gdn.norm, 16, 3, 128, workspace->projected_bf16_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
      } else {
        if (leaf_timings != nullptr) {
          leaf_timings->attention_prep_fused_with_core = true;
        }
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->attn_query_split);
        }
        if (error == cudaSuccess) {
          error = launch_split_attention_query_gate(
              workspace->projection_a_, 24, 256, workspace->gdn_convolved_,
              workspace->projection_b_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        const AttentionConfig config{
            24, 4, 256, 64, static_cast<std::uint32_t>(session->capacity_)};
        const std::size_t cache_stride =
            session->capacity_ * internal::kAttentionKvWidth;
        const AttentionCache committed{
            session->attention_key_ + attention_slot * cache_stride,
            session->attention_value_ + attention_slot * cache_stride};
        const AttentionCache candidate{
            workspace->attention_candidate_key_ +
                attention_slot * internal::kAttentionKvWidth,
            workspace->attention_candidate_value_ +
                attention_slot * internal::kAttentionKvWidth};
        if (error == cudaSuccess) {
          error = begin_phase(
              leaves, leaf_timings == nullptr
                          ? nullptr
                          : &leaf_timings->attn_qk_prep_softmax_pv_merge);
        }
        if (error == cudaSuccess) {
          const int n_parts =
              decode_kv_parts_for_position(session->frontier_);
          if (n_parts > 1) {
            error = launch_attention_prepare_partitioned(
                config, session->frontier_, workspace->gdn_convolved_,
                workspace->projection_c_, workspace->projection_d_,
                layer.attention.query_norm, layer.attention.key_norm,
                workspace->projection_b_, committed, candidate,
                workspace->attention_normalized_query_,
                workspace->attention_normalized_key_,
                workspace->attention_scores_, workspace->gdn_recurrent_output_,
                reinterpret_cast<float*>(workspace->prompt_projected_bf16_),
                reinterpret_cast<float*>(workspace->prompt_q8_), n_parts,
                nullptr);
          } else {
            error = launch_attention_prepare(
                config, session->frontier_, workspace->gdn_convolved_,
                workspace->projection_c_, workspace->projection_d_,
                layer.attention.query_norm, layer.attention.key_norm,
                workspace->projection_b_, committed, candidate,
                workspace->attention_normalized_query_,
                workspace->attention_normalized_key_,
                workspace->attention_scores_, workspace->gdn_recurrent_output_,
                nullptr);
          }
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->attn_output_cast);
        }
        if (error == cudaSuccess) {
          error = launch_fp32_to_bf16(
              workspace->gdn_recurrent_output_, internal::kAttentionQueryWidth,
              workspace->projected_bf16_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
      }
      if (exclusive && error == cudaSuccess) error = end_phase(categories);
    }
    {
      const NvtxRange mixer_range("qw38.mixer_mmv");
      if (exclusive && error == cudaSuccess) {
        error = begin_phase(categories, mixer_mmv);
      }
      if (gdn_layer) {
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_gdn_output);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.gdn.output, workspace->projected_bf16_,
                                workspace, workspace->mixer_output_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        ++gdn_slot;
      } else {
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_attn_output);
        }
        if (error == cudaSuccess) {
          error = matrix_vector(layer.attention.output,
                                workspace->projected_bf16_, workspace,
                                workspace->mixer_output_, nullptr);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        ++attention_slot;
      }
      {
        const NvtxRange core_range(core_nvtx);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->residual_mixer);
        }
        if (error == cudaSuccess) {
          residual_add_fp32<<<20, kThreads>>>(
              residual, workspace->mixer_output_, internal::kResidualWidth,
              next);
          error = cudaPeekAtLastError();
        }
        if (error == cudaSuccess) error = end_phase(leaves);
      }
      if (exclusive && error == cudaSuccess) error = end_phase(categories);
    }
    if (!exclusive && error == cudaSuccess) error = end_phase(categories);
    nvtxRangePop();
    nvtxRangePushA("qw38.ffn");
    if (error == cudaSuccess) {
      error = begin_phase(categories,
                          exclusive ? &decode_attribution->ffn_mmv
                                    : (timings == nullptr ? nullptr
                                                          : &timings->ffn));
    }
    if (error == cudaSuccess) {
      if (graphs != nullptr) {
        if (leaf_timings != nullptr) leaf_timings->ffn_graph_fused = true;
        const auto graph_started = std::chrono::steady_clock::now();
        nvtxRangePushA("qw38.graph_launch");
        error = cudaGraphLaunch(graphs->executions_[layer_index], nullptr);
        nvtxRangePop();
        const float graph_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - graph_started)
                .count());
        if (timings != nullptr) {
          timings->graph_launch.milliseconds += graph_ms;
          timings->graph_launch.measured = true;
        }
        if (decode_attribution != nullptr) {
          decode_attribution->graph.milliseconds += graph_ms;
          decode_attribution->graph.measured = true;
        }
      } else {
        const float* next_input_norm =
            layer_index + 1 < model.layers_.size()
                ? model.layers_[layer_index + 1].common.input_norm
                : nullptr;
        error = execute_ffn(layer.common, next, workspace, residual,
                            next_input_norm, pointwise_path, nullptr, leaves,
                            leaf_timings, capture, layer_index, layer_kind);
      }
    }
    if (error == cudaSuccess) error = end_phase(categories);
    nvtxRangePop();
#ifdef QW38_DIAGNOSTIC_TRACE
    if (error == cudaSuccess && active_trace != nullptr &&
        (layer_index == 0 || layer_index == 3 || layer_index == 63) &&
        internal::trace_filter_matches(*active_trace->filter, layer_index,
                                       "layer_residual")) {
      error = cudaDeviceSynchronize();
      if (error == cudaSuccess) {
        const Status trace_status = emit_cuda_trace(
            *active_trace->filter, active_trace->sink, active_trace->context,
            "layer_residual", layer_index, residual,
            internal::kResidualWidth, workspace->candidate_hidden_host_);
        if (!trace_status.is_ok()) return trace_status;
      }
    }
#endif
    if (error == cudaSuccess && control != nullptr &&
        control->poll != nullptr) {
      error = cudaDeviceSynchronize();
      if (error == cudaSuccess) {
        poll_status = control->poll(control->context);
        interrupted = !poll_status.is_ok();
      }
    }
  }
  if (error == cudaSuccess && !interrupted &&
      (gdn_slot != kGdnLayers || attention_slot != kAttentionLayers)) {
    error = cudaErrorInvalidValue;
  }
  nvtxRangePushA("qw38.logits");
  if (error == cudaSuccess && !interrupted) {
    error = begin_phase(categories,
                        decode_attribution == nullptr
                            ? (timings == nullptr ? nullptr : &timings->logits)
                            : &decode_attribution->logits);
  }
  if (error == cudaSuccess && !interrupted) {
    error = begin_phase(leaves, leaf_timings == nullptr
                                    ? nullptr
                                    : &leaf_timings->logits_norm);
  }
  if (error == cudaSuccess && !interrupted) {
    error = launch_rms_norm_fp32_dispatch(
        residual, model.output_norm_, internal::kResidualWidth,
        workspace->normalized_, nullptr);
  }
  if (error == cudaSuccess && !interrupted) error = end_phase(leaves);
  if (error == cudaSuccess && !interrupted && capture != nullptr) {
    error = cudaDeviceSynchronize();
    if (error == cudaSuccess) {
      error = capture_bf16_vector(
          workspace->normalized_, internal::kResidualWidth,
          capture->final_norm.data(), capture->final_norm_sha256,
          capture->final_norm_prefix.data(), capture->final_norm_prefix.size());
    }
    if (error == cudaSuccess) capture->final_norm_captured = true;
  }
  if (error == cudaSuccess && !interrupted) {
    error = begin_phase(leaves, leaf_timings == nullptr
                                    ? nullptr
                                    : &leaf_timings->logits_projection);
  }
  if (error == cudaSuccess && !interrupted) {
    error = matrix_vector(model.output_, workspace->normalized_, workspace,
                          workspace->logits_, nullptr);
  }
  if (error == cudaSuccess && !interrupted) error = end_phase(leaves);
#ifdef QW38_DIAGNOSTIC_TRACE
  if (error == cudaSuccess && !interrupted && active_trace != nullptr &&
      internal::trace_filter_matches(*active_trace->filter,
                                     internal::kTraceAllLayers, "final_norm")) {
    bf16_to_fp32<<<20, kThreads>>>(workspace->normalized_,
                                   internal::kResidualWidth,
                                   workspace->residual_b_);
    error = cudaPeekAtLastError();
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error == cudaSuccess) {
      const Status trace_status = emit_cuda_trace(
          *active_trace->filter, active_trace->sink, active_trace->context,
          "final_norm", internal::kTraceAllLayers, workspace->residual_b_,
          internal::kResidualWidth, workspace->candidate_hidden_host_);
      if (!trace_status.is_ok()) return trace_status;
    }
  }
#endif
  if (error == cudaSuccess && !interrupted) error = cudaEventRecord(stop);
  if (error == cudaSuccess && !interrupted) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess && !interrupted) {
    error = cudaEventElapsedTime(elapsed_milliseconds, start, stop);
  }
  if (error == cudaSuccess && !interrupted) {
    error = begin_phase(leaves, leaf_timings == nullptr ? nullptr
                                                       : &leaf_timings->d2h);
  }
  if (error == cudaSuccess && !interrupted) {
    error = cudaMemcpy(workspace->candidate_logits_host_, workspace->logits_,
                       logits_count * sizeof(float), cudaMemcpyDeviceToHost);
  }
#ifdef QW38_DIAGNOSTIC_TRACE
  if (error == cudaSuccess && !interrupted && active_trace != nullptr &&
      internal::trace_filter_matches(*active_trace->filter,
                                     internal::kTraceAllLayers, "logits")) {
    const Status trace_status = internal::emit_trace_tensor(
        *active_trace->filter, active_trace->sink, active_trace->context,
        {"logits", internal::kTraceAllLayers, workspace->candidate_logits_host_,
         logits_count, {logits_count, 0, 0}, 1});
    if (!trace_status.is_ok()) return trace_status;
  }
#endif
  if (error == cudaSuccess && !interrupted) {
    error = cudaMemcpy(workspace->candidate_hidden_host_, residual,
                       hidden_count * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess && !interrupted) error = end_phase(leaves);
  if (error == cudaSuccess && !interrupted && capture != nullptr) {
    write_float_sha(workspace->candidate_logits_host_, logits_count,
                    capture->output_sha256);
    copy_prefix(workspace->candidate_logits_host_, logits_count,
                capture->output_prefix.data(), capture->output_prefix.size());
    capture->output_count = logits_count;
    capture->output_captured = true;
  }
  if (error == cudaSuccess && !interrupted) error = end_phase(categories);
  nvtxRangePop();
  if (interrupted) {
    if (stop != nullptr) cudaEventDestroy(stop);
    if (start != nullptr) cudaEventDestroy(start);
    return poll_status;
  }
  if (error != cudaSuccess) {
    if (stop != nullptr) cudaEventDestroy(stop);
    if (start != nullptr) cudaEventDestroy(start);
    return cuda_status(error, "CUDA hybrid token schedule failed");
  }
  nvtxRangePushA("qw38.state_commit");
  error = begin_phase(categories,
                      decode_attribution == nullptr
                          ? (timings == nullptr ? nullptr : &timings->state_commit)
                          : &decode_attribution->state_commit);
  if (error == cudaSuccess) {
    error = begin_phase(leaves, leaf_timings == nullptr
                                    ? nullptr
                                    : &leaf_timings->state_copies);
  }
  const std::size_t cache_stride =
      session->capacity_ * internal::kAttentionKvWidth;
  const AttentionConfig commit_config{
      24, 4, 256, 64, static_cast<std::uint32_t>(session->capacity_)};
  error = launch_attention_scatter_layers(
      commit_config, session->frontier_, 1, kAttentionLayers,
      workspace->attention_candidate_key_, workspace->attention_candidate_value_,
      internal::kAttentionKvWidth, session->attention_key_,
      session->attention_value_, cache_stride, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) error = end_phase(leaves);
  if (error == cudaSuccess) error = end_phase(categories);
  nvtxRangePop();
  if (error != cudaSuccess) {
    if (stop != nullptr) cudaEventDestroy(stop);
    if (start != nullptr) cudaEventDestroy(start);
    return cuda_status(error, "CUDA hybrid token state commit failed");
  }
  if (error == cudaSuccess) error = end_phase(total);
  if (error == cudaSuccess && categories != nullptr) {
    error = categories->collect();
  }
  if (error == cudaSuccess && leaves != nullptr) error = leaves->collect();
  if (error == cudaSuccess && total != nullptr) error = total->collect();
  if (stop != nullptr) cudaEventDestroy(stop);
  if (start != nullptr) cudaEventDestroy(start);
  if (error != cudaSuccess) {
    return cuda_status(error, "cannot collect CUDA token attribution");
  }
  if (decode_attribution != nullptr) {
    const Status attribution_status =
        finish_decode_attribution(decode_attribution, attribution_started);
    if (!attribution_status.is_ok()) return attribution_status;
  }
  if (timings != nullptr && decode_attribution != nullptr) {
    timings->embedding = decode_attribution->embedding;
    timings->logits = decode_attribution->logits;
    timings->state_commit = decode_attribution->state_commit;
    timings->graph_launch = decode_attribution->graph;
    timings->gdn = decode_attribution->gdn_core;
    timings->attention = decode_attribution->attention_core;
    timings->ffn = decode_attribution->ffn_mmv;
  }
  if (timings != nullptr) {
    const float attributed = timings->embedding.milliseconds +
                             timings->gdn.milliseconds +
                             timings->attention.milliseconds +
                             timings->ffn.milliseconds +
                             timings->logits.milliseconds +
                             timings->state_commit.milliseconds;
    timings->idle_gaps = {
        std::max(0.0F, timings->token_total.milliseconds - attributed), true};
  }
  std::swap(session->gdn_convolution_,
            workspace->gdn_candidate_convolution_);
  std::swap(session->gdn_recurrent_, workspace->gdn_candidate_recurrent_);
  session->tokens_[session->frontier_] = token;
  std::memcpy(session->last_logits_, workspace->candidate_logits_host_,
              internal::kVocabularySize * sizeof(float));
  std::memcpy(session->last_hidden_, workspace->candidate_hidden_host_,
              internal::kResidualWidth * sizeof(float));
  std::memcpy(host_logits, workspace->candidate_logits_host_,
              internal::kVocabularySize * sizeof(float));
  std::memcpy(host_hidden, workspace->candidate_hidden_host_,
              internal::kResidualWidth * sizeof(float));
  ++session->frontier_;
  return Status::ok();
}

#ifdef QW38_DIAGNOSTIC_TRACE
Status execute_token_traced(const ResidentModel& model, std::size_t token,
                            SchedulerSession* session,
                            SchedulerWorkspace* workspace, float* host_logits,
                            std::size_t logits_count, float* host_hidden,
                            std::size_t hidden_count,
                            float* elapsed_milliseconds,
                            const internal::TraceFilter& filter,
                            internal::TraceSink sink, void* context) noexcept {
  Status status = internal::validate_trace_filter(filter);
  const bool layer_tap = std::strcmp(filter.tap, "layer_residual") == 0;
  const bool global_tap = std::strcmp(filter.tap, "final_norm") == 0 ||
                          std::strcmp(filter.tap, "logits") == 0;
  if (!status.is_ok() || sink == nullptr ||
      ((!layer_tap || (filter.layer != 0 && filter.layer != 3 &&
                       filter.layer != 63)) &&
       (!global_tap || filter.layer != internal::kTraceAllLayers))) {
    return {StatusCode::kInvalidArgument,
            "CUDA diagnostic trace filter is not an admitted boundary"};
  }
  TraceContext trace{&filter, sink, context};
  TraceContext* previous = active_trace;
  active_trace = &trace;
  status = execute_token(model, token, session, workspace, host_logits,
                         logits_count, host_hidden, hidden_count,
                         elapsed_milliseconds, nullptr, nullptr,
                         PointwisePath::kUnfused, nullptr);
  active_trace = previous;
  return status;
}
#endif

Status execute_prompt_chunk(
    const ResidentModel& model, const std::size_t* tokens,
    std::size_t token_count, SchedulerSession* session,
    SchedulerWorkspace* workspace, float* host_logits,
    std::size_t logits_count, float* host_hidden, std::size_t hidden_count,
    const EvalControl* control, PromptPipelinePath path,
    PromptPipelineCounters* counters, SchedulerGraphs* graphs,
    GdnScanPath gdn_scan, PrefillAttribution* attribution) noexcept {
  const bool fused = path == PromptPipelinePath::kFusedOverlapped;
  if (model.blob_ == nullptr || tokens == nullptr || token_count < 2 ||
      session == nullptr || workspace == nullptr || session->capacity_ == 0 ||
      token_count > workspace->prompt_chunk_rows_ ||
      token_count > session->capacity_ - session->frontier_ ||
      workspace->capacity_ != session->capacity_ || host_logits == nullptr ||
      logits_count != internal::kVocabularySize || host_hidden == nullptr ||
      hidden_count != internal::kResidualWidth ||
      (path != PromptPipelinePath::kFusedOverlapped &&
       path != PromptPipelinePath::kUnfusedSerial) ||
      (fused && (workspace->prompt_compute_stream_ == nullptr ||
                 workspace->prompt_copy_stream_ == nullptr ||
                 workspace->prompt_compute_done_ == nullptr)) ||
      (graphs != nullptr &&
       (path != PromptPipelinePath::kFusedOverlapped ||
        !graphs->matches(model, workspace)))) {
    return {StatusCode::kInvalidArgument,
            "CUDA prompt chunk input, state, or output is invalid"};
  }
  for (std::size_t row = 0; row < token_count; ++row) {
    if (tokens[row] >= internal::kVocabularySize) {
      return {StatusCode::kInvalidArgument,
              "CUDA prompt chunk contains an invalid token"};
    }
  }
  const NvtxRange chunk_range("qw38.prefill_chunk");
  GpuPhaseRecorder attribution_recorder;
  GpuPhaseRecorder leaf_recorder;
  GpuPhaseRecorder* categories =
      attribution == nullptr ? nullptr : &attribution_recorder;
  const bool record_leaves =
      attribution != nullptr && attribution->record_leaves;
  ActivationCapture* capture =
      attribution == nullptr ? nullptr : attribution->capture;
  GpuPhaseRecorder* leaves = record_leaves ? &leaf_recorder : nullptr;
  LeafTimings* leaf_timings =
      record_leaves ? &attribution->leaves : nullptr;
  if (capture != nullptr) {
    capture->position = session->frontier_ + token_count - 1;
  }
  cudaStream_t stream = fused ? workspace->prompt_compute_stream_ : nullptr;
  cudaError_t error = cudaSuccess;
  {
    const PdlScope pdl;
    const NvtxRange embedding_range("qw38.embedding");
    error = begin_phase(categories,
                         attribution == nullptr ? nullptr
                                                : &attribution->embedding,
                         stream);
    if (error == cudaSuccess) {
      error = begin_phase(leaves, leaf_timings == nullptr
                                      ? nullptr
                                      : &leaf_timings->embedding,
                          stream);
    }
    if (fused) {
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            reinterpret_cast<std::size_t*>(workspace->prompt_q8_), tokens,
            token_count * sizeof(std::size_t), cudaMemcpyHostToDevice, stream);
      }
      if (error == cudaSuccess) {
        error = launch_quant_rows_decode_widen(
            model.embedding_.kind, model.embedding_.data, model.embedding_.rows,
            model.embedding_.columns,
            reinterpret_cast<const std::size_t*>(workspace->prompt_q8_),
            token_count, workspace->prompt_residual_a_, stream);
        if (error == cudaSuccess) {
          bump(counters, &PromptPipelineCounters::embedding_kernel_launches);
        }
      }
    } else {
      for (std::size_t row = 0; error == cudaSuccess && row < token_count; ++row) {
        error = launch_quant_row_decode(
            model.embedding_.kind, model.embedding_.data, model.embedding_.rows,
            model.embedding_.columns, tokens[row],
            workspace->prompt_normalized_ + row * internal::kResidualWidth,
            stream);
        if (error == cudaSuccess) {
          bump(counters, &PromptPipelineCounters::embedding_kernel_launches);
        }
      }
      if (error == cudaSuccess) {
        error = quartz_launch_kernel(
            bf16_to_fp32,
            dim3(static_cast<unsigned int>(
                (token_count * internal::kResidualWidth + kThreads - 1) /
                kThreads)),
            dim3(kThreads), 0, stream, workspace->prompt_normalized_,
            token_count * internal::kResidualWidth,
            workspace->prompt_residual_a_);
        if (error == cudaSuccess) {
          bump(counters, &PromptPipelineCounters::widen_kernel_launches);
        }
      }
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) error = end_phase(categories);
  }
  float* residual = workspace->prompt_residual_a_;
  float* after_mixer = workspace->prompt_residual_b_;
  std::size_t gdn_slot = 0;
  std::size_t attention_slot = 0;
  Status poll_status = Status::ok();
  for (std::size_t layer_index = 0;
       error == cudaSuccess && poll_status.is_ok() &&
       layer_index < model.layers_.size();
       ++layer_index) {
    const DeviceLayer& layer = model.layers_[layer_index];
    const bool gdn_layer = layer.kind == internal::LayerKind::kGdn;
    const char* layer_kind = gdn_layer ? "gdn" : "attention";
    nvtxRangePushA(gdn_layer ? "qw38.gdn" : "qw38.attention");
    {
    const PdlScope pdl;
    TimingValue* mixer_mmq =
        attribution == nullptr ? nullptr : &attribution->mixer_mmq;
    TimingValue* core = attribution == nullptr
                            ? nullptr
                            : (gdn_layer ? &attribution->gdn_core
                                         : &attribution->attention_core);
    const char* core_nvtx =
        gdn_layer ? "qw38.gdn_core" : "qw38.attention_core";
    // Mixer RMSNorm feeds the input projections, so it cannot sit inside the
    // later core interval. Attribute it as an accumulating core phase when it
    // actually launches; fused layers after 0 skip this empty event pair.
    if (!fused || layer_index == 0) {
      const NvtxRange core_range(core_nvtx);
      if (error == cudaSuccess) {
        error = begin_phase(categories, core, stream);
      }
      if (error == cudaSuccess) {
        error = begin_phase(leaves, leaf_timings == nullptr
                                        ? nullptr
                                        : &leaf_timings->input_norm,
                            stream);
      }
      if (error == cudaSuccess) {
        error = launch_rms_norm_rows_fp32_to_bf16(
            residual, layer.common.input_norm, internal::kResidualWidth,
            token_count, workspace->prompt_normalized_, stream);
        if (error == cudaSuccess) {
          bump(counters, &PromptPipelineCounters::rms_norm_kernel_launches);
        }
      }
      if (error == cudaSuccess) error = end_phase(leaves);
      if (error == cudaSuccess) error = end_phase(categories);
    }
    if (error == cudaSuccess) {
      const __nv_bfloat16* row =
          workspace->prompt_normalized_ +
          (token_count - 1) * internal::kResidualWidth;
      error = maybe_capture_preprojection(capture, layer_index, true, layer_kind,
                                          row, internal::kResidualWidth);
    }
    {
      const NvtxRange mixer_range("qw38.mixer_mmq");
      if (error == cudaSuccess) {
        error = begin_phase(categories, mixer_mmq, stream);
      }
      if (error == cudaSuccess) {
        error = begin_phase(leaves,
                            leaf_timings == nullptr
                                ? nullptr
                                : &leaf_timings->activation_staging_mixer,
                            stream);
      }
      if (error == cudaSuccess) {
        error = launch_quantize_mmq_q8_1(
            QuantKind::kQ8_0, workspace->prompt_normalized_, token_count,
            internal::kResidualWidth, workspace->prompt_q8_, stream);
      }
      if (error == cudaSuccess) error = end_phase(leaves);
      if (gdn_layer) {
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_packed_qkv,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt_q8_quality_mma(
              layer.gdn.packed_qkv, workspace->prompt_q8_, token_count,
              workspace->prompt_projection_a_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_value_gate,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt_q8_quality_mma(
              layer.gdn.value_gate, workspace->prompt_q8_, token_count,
              workspace->prompt_projection_b_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_alpha,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt_q8_quality_mma(
              layer.gdn.alpha, workspace->prompt_q8_, token_count,
              workspace->prompt_projection_c_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_beta,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt_q8_quality_mma(
              layer.gdn.beta, workspace->prompt_q8_, token_count,
              workspace->prompt_projection_d_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
      } else {
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_query_gate,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt_q8_quality_mma(
              layer.attention.query_gate, workspace->prompt_q8_, token_count,
              workspace->prompt_projection_a_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_key,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt_q8_quality_mma(
              layer.attention.key, workspace->prompt_q8_, token_count,
              workspace->prompt_projection_c_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_value,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt_q8_quality_mma(
              layer.attention.value, workspace->prompt_q8_, token_count,
              workspace->prompt_projection_d_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
      }
      if (error == cudaSuccess) error = end_phase(categories);
    }
    {
      const NvtxRange core_range(core_nvtx);
      if (error == cudaSuccess) {
        error = begin_phase(categories, core, stream);
      }
      TimingValue* fused_core_leaf = nullptr;
      if (leaf_timings != nullptr) {
        if (gdn_layer) {
          leaf_timings->gdn_conv_fused_with_recurrence = true;
          leaf_timings->gdn_output_norm_fused_with_gate = true;
          fused_core_leaf = &leaf_timings->gdn_conv_qk_norm_recurrence;
        } else {
          leaf_timings->attention_prep_fused_with_core = true;
          fused_core_leaf = &leaf_timings->attn_qk_prep_softmax_pv_merge;
        }
      }
      if (error == cudaSuccess) {
        error = begin_phase(leaves, fused_core_leaf, stream);
      }
      if (gdn_layer) {
        if (error == cudaSuccess) {
          error = quartz_launch_kernel(
              prepare_gdn_gate_rows, dim3(static_cast<unsigned int>(token_count)),
              dim3(kThreads), 0, stream, workspace->prompt_projection_c_,
              workspace->prompt_projection_d_, layer.gdn.folded_a,
              layer.gdn.dt_bias, static_cast<std::size_t>(16),
              static_cast<std::size_t>(3), workspace->prompt_gdn_decay_,
              workspace->prompt_gdn_update_);
        }
        const GdnState committed{
            session->gdn_convolution_ +
                gdn_slot * internal::kGdnConvolutionValues,
            session->gdn_recurrent_ +
                gdn_slot * internal::kGdnRecurrentStateValues};
        const GdnState candidate{
            workspace->gdn_candidate_convolution_ +
                gdn_slot * internal::kGdnConvolutionValues,
            workspace->gdn_candidate_recurrent_ +
                gdn_slot * internal::kGdnRecurrentStateValues};
        bool used_fused_gate = false;
        if (error == cudaSuccess) {
          const std::size_t gdn_scratch_floats =
              workspace->prompt_chunk_rows_ * internal::kFfnWidth *
              sizeof(__nv_bfloat16) / sizeof(float);
          float* overlay = reinterpret_cast<float*>(
              workspace->prompt_projected_bf16_);
          GdnScanPath scan = gdn_scan;
          float* conv_out = gdn_fuses_conv() ? nullptr
                                            : workspace->prompt_gdn_convolved_;
          if (scan == GdnScanPath::kFusedTokenLoop) {
            if (gdn_fuses_gated_output()) {
              error = launch_gdn_quality_fused(
                  kGdnConfig, workspace->prompt_projection_a_,
                  layer.gdn.convolution, workspace->prompt_gdn_decay_,
                  workspace->prompt_gdn_update_, token_count, committed,
                  candidate, conv_out, workspace->prompt_gdn_recurrent_output_,
                  workspace->prompt_projection_b_, layer.gdn.norm,
                  workspace->prompt_projected_bf16_, stream, true, nullptr,
                  nullptr, nullptr, 0);
              if (error == cudaSuccess) used_fused_gate = true;
            } else {
              error = launch_gdn_prepare_chunk_tiled(
                  kGdnConfig, workspace->prompt_projection_a_,
                  layer.gdn.convolution, workspace->prompt_gdn_decay_,
                  workspace->prompt_gdn_update_, token_count, committed,
                  candidate, conv_out, workspace->prompt_gdn_recurrent_output_,
                  stream, scan, overlay, gdn_scratch_floats);
            }
            if (error == cudaErrorInvalidValue) {
              scan = GdnScanPath::kParallelAssociative;
              used_fused_gate = false;
            }
          }
          if (error == cudaSuccess || error == cudaErrorInvalidValue) {
            if (scan == GdnScanPath::kParallelAssociative &&
                gdn_scratch_floats < gdn_recurrent_values(kGdnConfig) +
                                         gdn_scan_operator_values(kGdnConfig)) {
              scan = GdnScanPath::kSequentialWindows;
            }
            if (scan != GdnScanPath::kFusedTokenLoop) {
              error = launch_gdn_prepare_chunk_tiled(
                  kGdnConfig, workspace->prompt_projection_a_,
                  layer.gdn.convolution, workspace->prompt_gdn_decay_,
                  workspace->prompt_gdn_update_, token_count, committed,
                  candidate, workspace->prompt_gdn_convolved_,
                  workspace->prompt_gdn_recurrent_output_, stream, scan,
                  overlay, gdn_scratch_floats);
            }
          }
        }
        if (error == cudaSuccess && !used_fused_gate) {
          error = launch_gdn_gated_output_rows_dispatch(
              workspace->prompt_gdn_recurrent_output_,
              workspace->prompt_projection_b_, layer.gdn.norm,
              static_cast<std::size_t>(16), static_cast<std::size_t>(3),
              static_cast<std::size_t>(128), token_count,
              workspace->prompt_projected_bf16_, stream);
        }
      } else {
        if (error == cudaSuccess) {
          const std::size_t values =
              token_count * internal::kAttentionQueryWidth;
          error = quartz_launch_kernel(
              split_attention_rows,
              dim3(static_cast<unsigned int>((values + kThreads - 1) / kThreads)),
              dim3(kThreads), 0, stream, workspace->prompt_projection_a_,
              internal::kAttentionQueryWidth, static_cast<std::size_t>(256),
              token_count, workspace->prompt_gdn_convolved_,
              workspace->prompt_projection_b_);
        }
        const AttentionConfig config{
            24, 4, 256, 64, static_cast<std::uint32_t>(session->capacity_)};
        const std::size_t cache_stride =
            session->capacity_ * internal::kAttentionKvWidth;
        const AttentionCache committed{
            session->attention_key_ + attention_slot * cache_stride,
            session->attention_value_ + attention_slot * cache_stride};
        const std::size_t candidate_stride =
            workspace->prompt_chunk_rows_ * internal::kAttentionKvWidth;
        const AttentionCache candidate{
            workspace->prompt_attention_candidate_key_ +
                attention_slot * candidate_stride,
            workspace->prompt_attention_candidate_value_ +
                attention_slot * candidate_stride};
        if (error == cudaSuccess) {
          if (fattn_uses_stream_k()) {
            const __half* prepared_q = nullptr;
            if (fattn_uses_prepared_query()) {
              const std::size_t need =
                  attention_prepared_query_bytes(config, token_count);
              const std::size_t overlay_bytes =
                  workspace->prompt_chunk_rows_ * kMaximumProjection *
                  sizeof(float);
              if (need == 0 || need > overlay_bytes) {
                error = cudaErrorInvalidValue;
              } else {
                prepared_q = reinterpret_cast<const __half*>(
                    workspace->prompt_projection_a_);
              }
            }
            if (error == cudaSuccess) {
              error = launch_attention_prepare_chunk_stream_k(
                  config, session->frontier_, token_count,
                  workspace->prompt_gdn_convolved_,
                  workspace->prompt_projection_c_,
                  workspace->prompt_projection_d_, layer.attention.query_norm,
                  layer.attention.key_norm, workspace->prompt_projection_b_,
                  committed, candidate, workspace->attention_normalized_query_,
                  workspace->attention_normalized_key_,
                  workspace->attention_scores_,
                  workspace->prompt_gdn_recurrent_output_,
                  reinterpret_cast<float*>(workspace->prompt_projected_bf16_),
                  reinterpret_cast<float*>(workspace->prompt_q8_), stream,
                  prepared_q);
            }
            // Persistent stream-K aliases prompt_q8_ for packed dst_tmp_meta
            // (fits; no extra cudaMalloc). prompt_projected_bf16_ remains the
            // OPT-026 partial_vkq alias when persistent is off. Prepared Q
            // aliases prompt_projection_a_ after split_attention_rows.
          } else {
            error = launch_attention_prepare_chunk(
                config, session->frontier_, token_count,
                workspace->prompt_gdn_convolved_,
                workspace->prompt_projection_c_,
                workspace->prompt_projection_d_, layer.attention.query_norm,
                layer.attention.key_norm, workspace->prompt_projection_b_,
                committed, candidate, workspace->attention_normalized_query_,
                workspace->attention_normalized_key_,
                workspace->attention_scores_,
                workspace->prompt_gdn_recurrent_output_, stream);
          }
        }
        if (error == cudaSuccess) {
          error = launch_fp32_to_bf16(
              workspace->prompt_gdn_recurrent_output_,
              token_count * internal::kAttentionQueryWidth,
              workspace->prompt_projected_bf16_, stream);
        }
      }
      if (error == cudaSuccess) error = end_phase(leaves);
      if (error == cudaSuccess) error = end_phase(categories);
    }
    {
      const NvtxRange mixer_range("qw38.mixer_mmq");
      if (error == cudaSuccess) {
        error = begin_phase(categories, mixer_mmq, stream);
      }
      if (gdn_layer) {
        if (error == cudaSuccess) {
          error = begin_phase(leaves,
                              leaf_timings == nullptr
                                  ? nullptr
                                  : &leaf_timings->activation_staging_mixer,
                              stream);
        }
        if (error == cudaSuccess) {
          error = launch_quantize_mmq_q8_1(
              QuantKind::kQ8_0, workspace->prompt_projected_bf16_, token_count,
              internal::kGdnValueWidth, workspace->prompt_q8_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_gdn_output,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt_q8_quality_mma(
              layer.gdn.output, workspace->prompt_q8_, token_count,
              workspace->prompt_mixer_output_, stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        ++gdn_slot;
      } else {
        if (error == cudaSuccess) {
          error = begin_phase(leaves, leaf_timings == nullptr
                                          ? nullptr
                                          : &leaf_timings->proj_attn_output,
                              stream);
        }
        if (error == cudaSuccess) {
          error = matrix_prompt(layer.attention.output,
                                workspace->prompt_projected_bf16_, token_count,
                                workspace, workspace->prompt_mixer_output_,
                                stream);
        }
        if (error == cudaSuccess) error = end_phase(leaves);
        ++attention_slot;
      }
      if (error == cudaSuccess) error = end_phase(categories);
    }
    }  // PdlScope: mixer + core + output mixer only
    nvtxRangePop();
    nvtxRangePushA("qw38.ffn");
    if (error == cudaSuccess) {
      error = begin_phase(categories,
                           attribution == nullptr ? nullptr
                                                  : &attribution->ffn_mmq,
                           stream);
    }
    const bool replay_prompt_ffn =
        graphs != nullptr &&
        graphs->prompt_graph_count() == internal::kModelLayerCount &&
        token_count == graphs->prompt_graph_rows();
    if (error == cudaSuccess) {
      if (replay_prompt_ffn) {
        if (leaf_timings != nullptr) leaf_timings->ffn_graph_fused = true;
        const auto graph_started = std::chrono::steady_clock::now();
        nvtxRangePushA("qw38.graph_launch");
        error = cudaGraphLaunch(graphs->prompt_executions_[layer_index], stream);
        nvtxRangePop();
        if (error == cudaSuccess) {
          bump(counters, &PromptPipelineCounters::prompt_graph_launches);
          if (attribution != nullptr) {
            attribution->graph.milliseconds += static_cast<float>(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - graph_started)
                    .count());
            attribution->graph.measured = true;
            ++attribution->prompt_graph_launches;
          }
        }
      } else if (fused) {
        const float* next_input_norm =
            layer_index + 1 < model.layers_.size()
                ? model.layers_[layer_index + 1].common.input_norm
                : nullptr;
        error = execute_prompt_ffn_attributed(
            layer.common, residual, workspace, after_mixer, residual,
            next_input_norm, token_count, stream, leaves, leaf_timings, capture,
            layer_index, layer_kind);
        if (error == cudaSuccess) {
          bump(counters,
               &PromptPipelineCounters::fused_residual_norm_kernel_launches);
          if (next_input_norm != nullptr) {
            bump(counters,
                 &PromptPipelineCounters::fused_residual_norm_kernel_launches);
          } else {
            bump(counters, &PromptPipelineCounters::residual_add_kernel_launches);
          }
        }
      } else {
        error = launch_residual_add_fp32(
            residual, workspace->prompt_mixer_output_,
            token_count * internal::kResidualWidth, after_mixer, stream);
        if (error == cudaSuccess) {
          bump(counters, &PromptPipelineCounters::residual_add_kernel_launches);
        }
        if (error == cudaSuccess) {
          error = launch_rms_norm_rows_fp32_to_bf16(
              after_mixer, layer.common.ffn_norm, internal::kResidualWidth,
              token_count, workspace->prompt_normalized_, stream);
          if (error == cudaSuccess) {
            bump(counters, &PromptPipelineCounters::rms_norm_kernel_launches);
          }
        }
        if (error == cudaSuccess) {
          error = execute_prompt_ffn_projections(
              layer.common, workspace, token_count, stream, leaves,
              leaf_timings);
        }
        if (error == cudaSuccess) {
          error = launch_residual_add_fp32(
              after_mixer, workspace->prompt_mixer_output_,
              token_count * internal::kResidualWidth, residual, stream);
          if (error == cudaSuccess) {
            bump(counters, &PromptPipelineCounters::residual_add_kernel_launches);
          }
        }
      }
    }
    if (error == cudaSuccess) error = end_phase(categories);
    nvtxRangePop();
    if (error == cudaSuccess && control != nullptr &&
        control->poll != nullptr) {
      if (fused) {
        error = cudaStreamSynchronize(stream);
        if (error == cudaSuccess) {
          bump(counters, &PromptPipelineCounters::stream_synchronizes);
        }
      } else {
        error = cudaDeviceSynchronize();
        if (error == cudaSuccess) {
          bump(counters, &PromptPipelineCounters::device_synchronizes);
        }
      }
      if (error == cudaSuccess) {
        poll_status = control->poll(control->context);
        bump(counters, &PromptPipelineCounters::layer_polls);
      }
    }
  }
  if (!poll_status.is_ok()) return poll_status;
  if (error == cudaSuccess &&
      (gdn_slot != kGdnLayers || attention_slot != kAttentionLayers)) {
    error = cudaErrorInvalidValue;
  }
  const float* final_hidden =
      residual + (token_count - 1) * internal::kResidualWidth;
  nvtxRangePushA("qw38.logits");
  {
    const PdlScope pdl;
  if (error == cudaSuccess) {
    error = begin_phase(categories,
                          attribution == nullptr ? nullptr : &attribution->logits,
                          stream);
  }
  if (error == cudaSuccess) {
    error = begin_phase(leaves, leaf_timings == nullptr
                                    ? nullptr
                                    : &leaf_timings->logits_norm,
                        stream);
  }
  if (error == cudaSuccess) {
    error = launch_rms_norm_fp32_dispatch(
        final_hidden, model.output_norm_, internal::kResidualWidth,
        workspace->normalized_, stream);
    if (error == cudaSuccess) {
      bump(counters, &PromptPipelineCounters::rms_norm_kernel_launches);
    }
  }
  if (error == cudaSuccess) error = end_phase(leaves);
  if (error == cudaSuccess && capture != nullptr) {
    error = capture_bf16_vector(
        workspace->normalized_, internal::kResidualWidth,
        capture->final_norm.data(), capture->final_norm_sha256,
        capture->final_norm_prefix.data(), capture->final_norm_prefix.size());
    if (error == cudaSuccess) capture->final_norm_captured = true;
  }
  if (error == cudaSuccess) {
    error = begin_phase(leaves, leaf_timings == nullptr
                                    ? nullptr
                                    : &leaf_timings->logits_projection,
                        stream);
  }
  if (error == cudaSuccess) {
    error = matrix_vector(model.output_, workspace->normalized_, workspace,
                          workspace->logits_, stream);
  }
  if (error == cudaSuccess) error = end_phase(leaves);
  if (error == cudaSuccess) error = end_phase(categories);
  }
  nvtxRangePop();
  const std::size_t cache_stride =
      session->capacity_ * internal::kAttentionKvWidth;
  const std::size_t candidate_stride =
      workspace->prompt_chunk_rows_ * internal::kAttentionKvWidth;
  const AttentionConfig commit_config{
      24, 4, 256, 64, static_cast<std::uint32_t>(session->capacity_)};
  nvtxRangePushA("qw38.state_commit");
  if (fused) {
    if (leaf_timings != nullptr) leaf_timings->d2h_overlaps_state_copies = true;
    if (error == cudaSuccess) {
      error = begin_phase(categories,
                            attribution == nullptr ? nullptr
                                                   : &attribution->commit_sync,
                            stream);
    }
    if (error == cudaSuccess) {
      error = begin_phase(leaves, leaf_timings == nullptr
                                      ? nullptr
                                      : &leaf_timings->state_copies,
                          stream);
    }
    if (error == cudaSuccess) {
      error = cudaEventRecord(workspace->prompt_compute_done_, stream);
    }
    if (error == cudaSuccess) {
      error = cudaStreamWaitEvent(workspace->prompt_copy_stream_,
                                  workspace->prompt_compute_done_, 0);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(
          workspace->candidate_logits_host_, workspace->logits_,
          logits_count * sizeof(float), cudaMemcpyDeviceToHost,
          workspace->prompt_copy_stream_);
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::async_d2h_copies);
      }
    }
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(
          workspace->candidate_hidden_host_, final_hidden,
          hidden_count * sizeof(float), cudaMemcpyDeviceToHost,
          workspace->prompt_copy_stream_);
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::async_d2h_copies);
      }
    }
    if (error == cudaSuccess) {
      error = launch_attention_scatter_layers(
          commit_config, session->frontier_, token_count, kAttentionLayers,
          workspace->prompt_attention_candidate_key_,
          workspace->prompt_attention_candidate_value_, candidate_stride,
          session->attention_key_, session->attention_value_, cache_stride,
          stream);
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::scatter_kernel_launches);
      }
    }
    if (error == cudaSuccess) {
      error = cudaStreamSynchronize(workspace->prompt_copy_stream_);
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::stream_synchronizes);
      }
    }
    if (error == cudaSuccess) {
      error = cudaStreamSynchronize(stream);
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::stream_synchronizes);
      }
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) error = end_phase(categories);
  } else {
    if (error == cudaSuccess) {
      error = begin_phase(leaves, leaf_timings == nullptr ? nullptr
                                                         : &leaf_timings->d2h,
                          stream);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(workspace->candidate_logits_host_, workspace->logits_,
                         logits_count * sizeof(float), cudaMemcpyDeviceToHost);
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::blocking_d2h_copies);
      }
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(workspace->candidate_hidden_host_, final_hidden,
                         hidden_count * sizeof(float), cudaMemcpyDeviceToHost);
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::blocking_d2h_copies);
      }
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) {
      error = begin_phase(categories,
                            attribution == nullptr ? nullptr
                                                   : &attribution->commit_sync,
                            stream);
    }
    if (error == cudaSuccess) {
      error = begin_phase(leaves, leaf_timings == nullptr
                                      ? nullptr
                                      : &leaf_timings->state_copies,
                          stream);
    }
    for (std::size_t slot = 0; error == cudaSuccess && slot < kAttentionLayers;
         ++slot) {
      const AttentionCache committed{
          session->attention_key_ + slot * cache_stride,
          session->attention_value_ + slot * cache_stride};
      const AttentionCache candidate{
          workspace->prompt_attention_candidate_key_ + slot * candidate_stride,
          workspace->prompt_attention_candidate_value_ +
              slot * candidate_stride};
      error = launch_attention_scatter_chunk(
          commit_config, session->frontier_, token_count, candidate, committed,
          stream);
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::scatter_kernel_launches);
      }
    }
    if (error == cudaSuccess) {
      error = cudaDeviceSynchronize();
      if (error == cudaSuccess) {
        bump(counters, &PromptPipelineCounters::device_synchronizes);
      }
    }
    if (error == cudaSuccess) error = end_phase(leaves);
    if (error == cudaSuccess) error = end_phase(categories);
  }
  nvtxRangePop();
  if (error != cudaSuccess) {
    if (fused) {
      if (workspace->prompt_copy_stream_ != nullptr) {
        cudaStreamSynchronize(workspace->prompt_copy_stream_);
      }
      if (workspace->prompt_compute_stream_ != nullptr) {
        cudaStreamSynchronize(workspace->prompt_compute_stream_);
      }
    }
    return cuda_status(error, "CUDA hybrid prompt chunk failed");
  }
  if (categories != nullptr) {
    error = categories->collect();
    if (error == cudaSuccess) error = categories->reset();
    if (error != cudaSuccess) {
      return cuda_status(error, "cannot collect CUDA prefill attribution");
    }
  }
  if (leaves != nullptr) {
    error = leaves->collect();
    if (error == cudaSuccess) error = leaves->reset();
    if (error != cudaSuccess) {
      return cuda_status(error, "cannot collect CUDA prefill leaf attribution");
    }
  }
  if (capture != nullptr && error == cudaSuccess) {
    write_float_sha(workspace->candidate_logits_host_, logits_count,
                    capture->output_sha256);
    copy_prefix(workspace->candidate_logits_host_, logits_count,
                capture->output_prefix.data(), capture->output_prefix.size());
    capture->output_count = logits_count;
    capture->output_captured = true;
  }
  std::swap(session->gdn_convolution_,
            workspace->gdn_candidate_convolution_);
  std::swap(session->gdn_recurrent_, workspace->gdn_candidate_recurrent_);
  std::memcpy(session->tokens_ + session->frontier_, tokens,
              token_count * sizeof(std::size_t));
  std::memcpy(session->last_logits_, workspace->candidate_logits_host_,
              logits_count * sizeof(float));
  std::memcpy(session->last_hidden_, workspace->candidate_hidden_host_,
              hidden_count * sizeof(float));
  std::memcpy(host_logits, workspace->candidate_logits_host_,
              logits_count * sizeof(float));
  std::memcpy(host_hidden, workspace->candidate_hidden_host_,
              hidden_count * sizeof(float));
  session->frontier_ += token_count;
  return Status::ok();
}

Status greedy_sample(const SchedulerSession& session,
                     std::size_t* token, RuntimeTimings* timings) noexcept {
  const NvtxRange range("qw38.sampling");
  const auto started = std::chrono::steady_clock::now();
  if (token == nullptr || session.frontier_ == 0 ||
      session.last_logits_ == nullptr) {
    return {StatusCode::kInvalidArgument,
            "CUDA greedy sampling input or committed logits are invalid"};
  }
  std::size_t best = 0;
  for (std::size_t index = 1; index < internal::kVocabularySize; ++index) {
    if (session.last_logits_[index] > session.last_logits_[best]) best = index;
  }
  *token = best;
  if (timings != nullptr) {
    timings->sampling = {
        static_cast<float>(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count()),
        true};
  }
  return Status::ok();
}

Status sync_tokens(const ResidentModel& model, const std::size_t* tokens,
                   std::size_t token_count, SchedulerSession* session,
                   SchedulerWorkspace* workspace, float* host_logits,
                   std::size_t logits_count, float* host_hidden,
                   std::size_t hidden_count, SyncResult* result,
                   const EvalControl* control, SchedulerGraphs* graphs,
                   GdnScanPath gdn_scan,
                   PrefillAttribution* attribution) noexcept {
  if (session == nullptr || workspace == nullptr || result == nullptr ||
      model.resident_bytes() == 0 || session->capacity_ == 0 ||
      workspace->capacity_ != session->capacity_ ||
      token_count > session->capacity_ ||
      (token_count > 0 &&
       (tokens == nullptr || host_logits == nullptr ||
        logits_count != internal::kVocabularySize || host_hidden == nullptr ||
        hidden_count != internal::kResidualWidth)) ||
      (token_count == 0 &&
       (tokens != nullptr || host_logits != nullptr || logits_count != 0 ||
        host_hidden != nullptr || hidden_count != 0))) {
    return {StatusCode::kInvalidArgument,
            "CUDA token synchronization input, state, or output is invalid"};
  }
  if (attribution != nullptr) {
    const bool record_leaves = attribution->record_leaves;
    ActivationCapture* capture = attribution->capture;
    *attribution = {};
    attribution->prompt_tokens = token_count;
    attribution->record_leaves = record_leaves;
    attribution->capture = capture;
  }
  for (std::size_t index = 0; index < token_count; ++index) {
    if (tokens[index] >= internal::kVocabularySize) {
      return {StatusCode::kInvalidArgument,
              "CUDA token synchronization contains an invalid token"};
    }
  }
  std::size_t common_prefix = 0;
  while (common_prefix < session->frontier_ &&
         common_prefix < token_count &&
         session->tokens_[common_prefix] == tokens[common_prefix]) {
    ++common_prefix;
  }
  const bool append_or_same = common_prefix == session->frontier_;
  const std::size_t start = append_or_same ? session->frontier_ : 0;
  *result = {common_prefix, append_or_same ? session->frontier_ : 0,
             token_count - start, !append_or_same};
  if (!append_or_same) {
    const Status reset_status = session->reset();
    if (!reset_status.is_ok()) return reset_status;
  }
  if (start == token_count && token_count > 0) {
    std::memcpy(host_logits, session->last_logits_,
                internal::kVocabularySize * sizeof(float));
    std::memcpy(host_hidden, session->last_hidden_,
                internal::kResidualWidth * sizeof(float));
    return Status::ok();
  }
  const auto wall_started = std::chrono::steady_clock::now();
  float elapsed = 0.0F;
  for (std::size_t index = start; index < token_count;) {
    const std::size_t remaining = token_count - index;
    const std::size_t chunk =
        std::min(workspace->prompt_chunk_rows_, remaining);
    const Status status =
        chunk == 1
            ? execute_token(model, tokens[index], session, workspace,
                            host_logits, logits_count, host_hidden,
                            hidden_count, &elapsed, control, nullptr,
                            PointwisePath::kFused, graphs)
            : execute_prompt_chunk(model, tokens + index, chunk, session,
                                   workspace, host_logits, logits_count,
                                   host_hidden, hidden_count, control,
                                   PromptPipelinePath::kFusedOverlapped,
                                   nullptr, graphs, gdn_scan, attribution);
    if (!status.is_ok()) return status;
    if (attribution != nullptr) {
      attribution->evaluated_tokens += chunk;
      ++attribution->chunk_count;
    }
    index += chunk;
  }
  return finish_prefill_attribution(attribution, wall_started);
}

}  // namespace qw38::cuda
