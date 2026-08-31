#include "full_scheduler.h"

#include <limits>
#include <utility>

#include <cuda_fp16.h>

#include "attention_decode.h"
#include "gdn_step.h"
#include "mixer.h"
#include "scheduler.h"
#include "scheduler_primitives.h"

namespace qw38::cuda {
namespace {

constexpr std::size_t kGdnLayers = 48;
constexpr std::size_t kAttentionLayers = 16;
constexpr std::size_t kMaximumProjection = internal::kFfnWidth;
constexpr std::size_t kTraceTapCount = 4;
constexpr GdnConfig kGdnConfig{16, 48, 128, 128, 4};
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

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
}

__global__ void residual_add_fp32(const float* residual,
                                  const float* correction,
                                  std::size_t count, float* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __fadd_rn(residual[index], correction[index]);
  }
}

Status cuda_status(cudaError_t error, const char* message) noexcept {
  return error == cudaSuccess
             ? Status::ok()
             : Status{StatusCode::kInternal, message};
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
                          float* output) noexcept {
  if (matrix.kind == QuantKind::kQ8_0) {
    const unsigned int blocks = static_cast<unsigned int>(
        (matrix.rows + (kThreads / kWarpSize) - 1) /
        (kThreads / kWarpSize));
    q8_mmv_bf16<<<blocks, kThreads>>>(matrix.data, matrix.rows, matrix.columns,
                                     activation, output);
    return cudaPeekAtLastError();
  }
  return launch_quant_mmv(matrix.kind, matrix.data, matrix.rows,
                          matrix.columns, activation, workspace->q8_, output,
                          nullptr);
}

cudaError_t execute_ffn(const DeviceCommonLayer& layer,
                        const float* residual,
                        SchedulerWorkspace* workspace,
                        float* output) noexcept {
  rms_norm_fp32_to_bf16<<<1, kThreads>>>(
      residual, layer.ffn_norm, internal::kResidualWidth,
      workspace->normalized_);
  cudaError_t error = cudaPeekAtLastError();
  if (error == cudaSuccess) {
    error = matrix_vector(layer.ffn_gate, workspace->normalized_, workspace,
                          workspace->projection_a_);
  }
  if (error == cudaSuccess) {
    error = matrix_vector(layer.ffn_up, workspace->normalized_, workspace,
                          workspace->projection_b_);
  }
  if (error == cudaSuccess) {
    error = launch_swiglu_bf16(
        workspace->projection_a_, workspace->projection_b_,
        internal::kFfnWidth, workspace->ffn_activated_, nullptr);
  }
  if (error == cudaSuccess) {
    error = matrix_vector(layer.ffn_down, workspace->ffn_activated_, workspace,
                          workspace->mixer_output_);
  }
  if (error == cudaSuccess) {
    residual_add_fp32<<<20, kThreads>>>(
        residual, workspace->mixer_output_, internal::kResidualWidth, output);
    error = cudaPeekAtLastError();
  }
  return error;
}

}  // namespace

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
  capacity_ = other.capacity_;
  frontier_ = other.frontier_;
  allocated_bytes_ = other.allocated_bytes_;
  other.gdn_convolution_ = nullptr;
  other.gdn_recurrent_ = nullptr;
  other.attention_key_ = nullptr;
  other.attention_value_ = nullptr;
  other.capacity_ = 0;
  other.frontier_ = 0;
  other.allocated_bytes_ = 0;
  return *this;
}

void SchedulerSession::release() noexcept {
  if (attention_value_ != nullptr) cudaFree(attention_value_);
  if (attention_key_ != nullptr) cudaFree(attention_key_);
  if (gdn_recurrent_ != nullptr) cudaFree(gdn_recurrent_);
  if (gdn_convolution_ != nullptr) cudaFree(gdn_convolution_);
  gdn_convolution_ = nullptr;
  gdn_recurrent_ = nullptr;
  attention_key_ = nullptr;
  attention_value_ = nullptr;
  capacity_ = 0;
  frontier_ = 0;
  allocated_bytes_ = 0;
}

Status SchedulerSession::create(std::size_t capacity) noexcept {
  if (capacity == 0 || capacity > 131072 || capacity_ != 0) {
    return {StatusCode::kInvalidArgument,
            "CUDA scheduler session capacity is invalid"};
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

std::size_t SchedulerSession::capacity() const noexcept { return capacity_; }
std::size_t SchedulerSession::frontier() const noexcept { return frontier_; }
std::size_t SchedulerSession::allocated_bytes() const noexcept {
  return allocated_bytes_;
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
  QW38_MOVE_POINTER(trace_taps_);
#undef QW38_MOVE_POINTER
  capacity_ = other.capacity_;
  allocated_bytes_ = other.allocated_bytes_;
  other.capacity_ = 0;
  other.allocated_bytes_ = 0;
  return *this;
}

void SchedulerWorkspace::release() noexcept {
#define QW38_FREE(name) if (name != nullptr) cudaFree(name); name = nullptr
  QW38_FREE(trace_taps_);
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
  capacity_ = 0;
  allocated_bytes_ = 0;
}

Status SchedulerWorkspace::create(std::size_t capacity) noexcept {
  if (capacity == 0 || capacity > 131072 || capacity_ != 0) {
    return {StatusCode::kInvalidArgument,
            "CUDA scheduler workspace capacity is invalid"};
  }
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
                internal::kGdnConvolutionValues);
  QW38_ALLOCATE(gdn_candidate_recurrent_,
                internal::kGdnRecurrentStateValues);
  QW38_ALLOCATE(attention_candidate_key_, internal::kAttentionKvWidth);
  QW38_ALLOCATE(attention_candidate_value_, internal::kAttentionKvWidth);
  QW38_ALLOCATE(attention_normalized_query_, internal::kAttentionQueryWidth);
  QW38_ALLOCATE(attention_normalized_key_, internal::kAttentionKvWidth);
  QW38_ALLOCATE(attention_scores_, 24 * capacity);
  QW38_ALLOCATE(logits_, internal::kVocabularySize);
  QW38_ALLOCATE(trace_taps_, kTraceTapCount * internal::kResidualWidth);
#undef QW38_ALLOCATE
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

Status execute_token(const ResidentModel& model, std::size_t token,
                     SchedulerSession* session, SchedulerWorkspace* workspace,
                     float* host_logits, std::size_t logits_count,
                     float* host_hidden, std::size_t hidden_count,
                     float* elapsed_milliseconds) noexcept {
  if (model.blob_ == nullptr || token >= internal::kVocabularySize ||
      session == nullptr || workspace == nullptr ||
      session->capacity_ == 0 || session->frontier_ >= session->capacity_ ||
      workspace->capacity_ != session->capacity_ || host_logits == nullptr ||
      logits_count != internal::kVocabularySize || host_hidden == nullptr ||
      hidden_count != internal::kResidualWidth ||
      elapsed_milliseconds == nullptr) {
    return {StatusCode::kInvalidArgument,
            "CUDA token scheduler input, state, or output is invalid"};
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
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
  std::size_t gdn_slot = 0;
  std::size_t attention_slot = 0;
  float* residual = workspace->residual_a_;
  float* next = workspace->residual_b_;
  for (std::size_t layer_index = 0;
       error == cudaSuccess && layer_index < model.layers_.size();
       ++layer_index) {
    const DeviceLayer& layer = model.layers_[layer_index];
    rms_norm_fp32_to_bf16<<<1, kThreads>>>(
        residual, layer.common.input_norm, internal::kResidualWidth,
        workspace->normalized_);
    error = cudaPeekAtLastError();
    if (layer.kind == internal::LayerKind::kGdn) {
      if (error == cudaSuccess) {
        error = matrix_vector(layer.gdn.packed_qkv, workspace->normalized_,
                              workspace, workspace->projection_a_);
      }
      if (error == cudaSuccess) {
        error = matrix_vector(layer.gdn.value_gate, workspace->normalized_,
                              workspace, workspace->projection_b_);
      }
      if (error == cudaSuccess) {
        error = matrix_vector(layer.gdn.alpha, workspace->normalized_,
                              workspace, workspace->projection_c_);
      }
      if (error == cudaSuccess) {
        error = matrix_vector(layer.gdn.beta, workspace->normalized_, workspace,
                              workspace->projection_d_);
      }
      if (error == cudaSuccess) {
        error = launch_prepare_gdn_gates(
            workspace->projection_c_, workspace->projection_d_,
            layer.gdn.folded_a, layer.gdn.dt_bias, 16, 3,
            workspace->gdn_decay_, workspace->gdn_update_, nullptr);
      }
      const GdnState committed{
          session->gdn_convolution_ +
              gdn_slot * internal::kGdnConvolutionValues,
          session->gdn_recurrent_ +
              gdn_slot * internal::kGdnRecurrentStateValues};
      const GdnState candidate{workspace->gdn_candidate_convolution_,
                               workspace->gdn_candidate_recurrent_};
      if (error == cudaSuccess) {
        error = launch_gdn_prepare_tiled(
            kGdnConfig, workspace->projection_a_, layer.gdn.convolution,
            workspace->gdn_decay_, workspace->gdn_update_, committed,
            candidate, workspace->gdn_convolved_,
            workspace->gdn_recurrent_output_, nullptr);
      }
      if (error == cudaSuccess) {
        error = launch_gdn_gated_output(
            workspace->gdn_recurrent_output_, workspace->projection_b_,
            layer.gdn.norm, 16, 3, 128, workspace->projected_bf16_, nullptr);
      }
      if (error == cudaSuccess) {
        error = matrix_vector(layer.gdn.output, workspace->projected_bf16_,
                              workspace, workspace->mixer_output_);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            committed.convolution, candidate.convolution,
            internal::kGdnConvolutionValues * sizeof(float),
            cudaMemcpyDeviceToDevice);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            committed.recurrent, candidate.recurrent,
            internal::kGdnRecurrentStateValues * sizeof(float),
            cudaMemcpyDeviceToDevice);
      }
      ++gdn_slot;
    } else {
      if (error == cudaSuccess) {
        error = matrix_vector(layer.attention.query_gate,
                              workspace->normalized_, workspace,
                              workspace->projection_a_);
      }
      if (error == cudaSuccess) {
        error = matrix_vector(layer.attention.key, workspace->normalized_,
                              workspace, workspace->projection_c_);
      }
      if (error == cudaSuccess) {
        error = matrix_vector(layer.attention.value, workspace->normalized_,
                              workspace, workspace->projection_d_);
      }
      if (error == cudaSuccess) {
        error = launch_split_attention_query_gate(
            workspace->projection_a_, 24, 256, workspace->gdn_convolved_,
            workspace->projection_b_, nullptr);
      }
      const AttentionConfig config{24, 4, 256, 64,
                                   static_cast<std::uint32_t>(session->capacity_)};
      const std::size_t cache_stride =
          session->capacity_ * internal::kAttentionKvWidth;
      const AttentionCache committed{
          session->attention_key_ + attention_slot * cache_stride,
          session->attention_value_ + attention_slot * cache_stride};
      const AttentionCache candidate{workspace->attention_candidate_key_,
                                     workspace->attention_candidate_value_};
      if (error == cudaSuccess) {
        error = launch_attention_prepare(
            config, session->frontier_, workspace->gdn_convolved_,
            workspace->projection_c_, workspace->projection_d_,
            layer.attention.query_norm, layer.attention.key_norm,
            workspace->projection_b_, committed, candidate,
            workspace->attention_normalized_query_,
            workspace->attention_normalized_key_, workspace->attention_scores_,
            workspace->gdn_recurrent_output_, nullptr);
      }
      if (error == cudaSuccess) {
        error = launch_fp32_to_bf16(
            workspace->gdn_recurrent_output_, internal::kAttentionQueryWidth,
            workspace->projected_bf16_, nullptr);
      }
      if (error == cudaSuccess) {
        error = matrix_vector(layer.attention.output,
                              workspace->projected_bf16_, workspace,
                              workspace->mixer_output_);
      }
      const std::size_t target =
          session->frontier_ * internal::kAttentionKvWidth;
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            committed.key + target, candidate.key,
            internal::kAttentionKvWidth * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            committed.value + target, candidate.value,
            internal::kAttentionKvWidth * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice);
      }
      ++attention_slot;
    }
    if (error == cudaSuccess) {
      residual_add_fp32<<<20, kThreads>>>(
          residual, workspace->mixer_output_, internal::kResidualWidth, next);
      error = cudaPeekAtLastError();
    }
    if (error == cudaSuccess) {
      error = execute_ffn(layer.common, next, workspace, residual);
    }
    std::size_t tap = kTraceTapCount;
    if (layer_index == 0) tap = 0;
    if (layer_index == 3) tap = 1;
    if (layer_index == 63) tap = 2;
    if (error == cudaSuccess && tap < kTraceTapCount) {
      error = cudaMemcpyAsync(
          workspace->trace_taps_ + tap * internal::kResidualWidth, residual,
          internal::kResidualWidth * sizeof(float), cudaMemcpyDeviceToDevice);
    }
  }
  if (error == cudaSuccess &&
      (gdn_slot != kGdnLayers || attention_slot != kAttentionLayers)) {
    error = cudaErrorInvalidValue;
  }
  if (error == cudaSuccess) {
    rms_norm_fp32_to_bf16<<<1, kThreads>>>(
        residual, model.output_norm_, internal::kResidualWidth,
        workspace->normalized_);
    error = cudaPeekAtLastError();
  }
  if (error == cudaSuccess) {
    error = matrix_vector(model.output_, workspace->normalized_, workspace,
                          workspace->logits_);
  }
  if (error == cudaSuccess) {
    bf16_to_fp32<<<20, kThreads>>>(
        workspace->normalized_, internal::kResidualWidth,
        workspace->trace_taps_ + 3 * internal::kResidualWidth);
    error = cudaPeekAtLastError();
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) {
    error = cudaEventElapsedTime(elapsed_milliseconds, start, stop);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_logits, workspace->logits_,
                       logits_count * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_hidden, residual,
                       hidden_count * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (stop != nullptr) cudaEventDestroy(stop);
  if (start != nullptr) cudaEventDestroy(start);
  if (error != cudaSuccess) {
    return cuda_status(error, "CUDA hybrid token schedule failed");
  }
  ++session->frontier_;
  return Status::ok();
}

}  // namespace qw38::cuda
