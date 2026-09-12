#include "gdn_step.h"
#include "gdn_fused_quality.cuh"
#include "gdn_decode_path.cuh"
#include "pdl_launch.cuh"

#include <algorithm>
#include <cmath>

QW38_PDL_REGISTER_DEVICE_OPS()

namespace qw38::cuda {
namespace {

constexpr std::uint32_t kMaximumKeyHeads = 16;
constexpr std::uint32_t kMaximumValueHeads = 48;
constexpr std::uint32_t kMaximumWidth = 128;
constexpr std::uint32_t kMaximumConvolutionWidth = 4;
constexpr int kThreads = 128;
constexpr int kConvolutionThreads = 256;
constexpr std::size_t kScanWindow = 64;
constexpr float kL2Epsilon = 1.0e-6F;

#include "gdn_decode_recurrence.cuh"

bool valid_config(const GdnConfig& config) noexcept {
  return config.key_heads > 0 && config.key_heads <= kMaximumKeyHeads &&
         config.value_heads > 0 && config.value_heads <= kMaximumValueHeads &&
         config.value_heads % config.key_heads == 0 && config.key_width > 0 &&
         config.key_width <= kMaximumWidth && config.value_width > 0 &&
         config.value_width <= kMaximumWidth &&
         config.convolution_width > 0 &&
         config.convolution_width <= kMaximumConvolutionWidth;
}

__global__ void prepare_convolution_window(
    const float* input, const float* weights, const float* source,
    float* candidate, float* output, std::size_t channels, std::uint32_t width,
    std::size_t token_count) {
  const std::size_t channel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  const std::size_t base = channel * width;
  float history[kMaximumConvolutionWidth];
  for (std::uint32_t index = 0; index < width; ++index) {
    history[index] = source[base + index];
  }
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::uint32_t index = 0; index + 1 < width; ++index) {
      history[index] = history[index + 1];
    }
    history[width - 1] = input[token * channels + channel];
    float convolution = 0.0F;
    for (std::uint32_t index = 0; index < width; ++index) {
      convolution = __fadd_rn(
          convolution, __fmul_rn(history[index], weights[base + index]));
    }
    output[token * channels + channel] =
        convolution / (1.0F + expf(-convolution));
  }
  for (std::uint32_t index = 0; index < width; ++index) {
    candidate[base + index] = history[index];
  }
}

__global__ void prepare_convolution_chunk_parallel(
    const float* input, const float* weights, const float* source,
    float* candidate, float* output, std::size_t channels, std::uint32_t width,
    std::size_t token_count) {
  quartz_pdl_sync();
  const std::size_t channel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t token = static_cast<std::size_t>(blockIdx.y);
  if (channel >= channels || token >= token_count) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t base = channel * width;
  float history[kMaximumConvolutionWidth];
  for (std::uint32_t index = 0; index < width; ++index) {
    history[index] =
        (token + index + 1 >= width)
            ? input[(token - (width - 1 - index)) * channels + channel]
            : source[base + index + token + 1];
  }
  float convolution = 0.0F;
  for (std::uint32_t index = 0; index < width; ++index) {
    convolution = __fadd_rn(convolution,
                              __fmul_rn(history[index], weights[base + index]));
  }
  output[token * channels + channel] = convolution / (1.0F + expf(-convolution));
  if (token + 1 == token_count) {
    for (std::uint32_t index = 0; index < width; ++index) {
      candidate[base + index] = history[index];
    }
  }
  quartz_pdl_lc();
}

__global__ void prepare_recurrence_window(
    GdnConfig config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, std::size_t token_count,
    bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (value_head >= config.value_heads || lane >= config.value_width) return;
  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::uint32_t key_head = value_head / reuse;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  __shared__ float query_inverse;
  __shared__ float key_inverse;
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * config.key_width *
      config.value_width;
  const std::size_t channels = 2 * query_count +
                               static_cast<std::size_t>(config.value_heads) *
                                   config.value_width;
  for (std::size_t token = 0; token < token_count; ++token) {
    const float* token_convolution = convolution_output + token * channels;
    const float* query = token_convolution + key_head * config.key_width;
    const float* key = token_convolution + query_count +
                       key_head * config.key_width;
    const std::uint32_t replica = value_head % reuse;
    const std::uint32_t tiled_head = replica * config.key_heads + key_head;
    const std::uint32_t source_head = value_is_tiled ? tiled_head : value_head;
    const float* value = token_convolution + 2 * query_count +
                         source_head * config.value_width;
    if (lane == 0) {
      float query_squares = 0.0F;
      float key_squares = 0.0F;
      for (std::uint32_t index = 0; index < config.key_width; ++index) {
        query_squares = __fadd_rn(
            query_squares, __fmul_rn(query[index], query[index]));
        key_squares =
            __fadd_rn(key_squares, __fmul_rn(key[index], key[index]));
      }
      query_inverse =
          1.0F / sqrtf(query_squares + kL2Epsilon) /
          sqrtf(static_cast<float>(config.key_width));
      key_inverse = 1.0F / sqrtf(key_squares + kL2Epsilon);
    }
    __syncthreads();

    const float* current = token == 0 ? source : candidate;
    const float decay =
        expf(log_decay[token * config.value_heads + value_head]);
    float prediction = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          lane;
      const float decayed = __fmul_rn(current[index], decay);
      prediction = __fadd_rn(
          prediction,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), decayed));
    }
    const float delta = __fmul_rn(
        value[lane] - prediction,
        beta[token * config.value_heads + value_head]);
    float result = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          lane;
      const float decayed = __fmul_rn(current[index], decay);
      const float updated = __fadd_rn(
          decayed,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), delta));
      candidate[index] = updated;
      result = __fadd_rn(
          result,
          __fmul_rn(__fmul_rn(query[key_lane], query_inverse), updated));
    }
    output[(token * config.value_heads + value_head) * config.value_width +
           lane] = result;
    __syncthreads();
  }
}

__global__ void prepare_recurrence_fused_token_loop(
    GdnConfig config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, std::size_t token_count,
    bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (value_head >= config.value_heads || lane >= config.value_width) return;
  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::uint32_t key_head = value_head / reuse;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  __shared__ float query_inverse;
  __shared__ float key_inverse;
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * config.key_width *
      config.value_width;
  const std::size_t channels = 2 * query_count +
                               static_cast<std::size_t>(config.value_heads) *
                                   config.value_width;
  float state[kMaximumWidth];
  for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
    const std::size_t index =
        head_base + static_cast<std::size_t>(key_lane) * config.value_width +
        lane;
    state[key_lane] = source[index];
  }
  for (std::size_t token = 0; token < token_count; ++token) {
    const float* token_convolution = convolution_output + token * channels;
    const float* query = token_convolution + key_head * config.key_width;
    const float* key = token_convolution + query_count +
                       key_head * config.key_width;
    const std::uint32_t replica = value_head % reuse;
    const std::uint32_t tiled_head = replica * config.key_heads + key_head;
    const std::uint32_t source_head = value_is_tiled ? tiled_head : value_head;
    const float* value = token_convolution + 2 * query_count +
                         source_head * config.value_width;
    if (lane == 0) {
      float query_squares = 0.0F;
      float key_squares = 0.0F;
      for (std::uint32_t index = 0; index < config.key_width; ++index) {
        query_squares = __fadd_rn(
            query_squares, __fmul_rn(query[index], query[index]));
        key_squares =
            __fadd_rn(key_squares, __fmul_rn(key[index], key[index]));
      }
      query_inverse =
          1.0F / sqrtf(query_squares + kL2Epsilon) /
          sqrtf(static_cast<float>(config.key_width));
      key_inverse = 1.0F / sqrtf(key_squares + kL2Epsilon);
    }
    __syncthreads();

    const float decay =
        expf(log_decay[token * config.value_heads + value_head]);
    float prediction = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const float decayed = __fmul_rn(state[key_lane], decay);
      prediction = __fadd_rn(
          prediction,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), decayed));
    }
    const float delta = __fmul_rn(
        value[lane] - prediction,
        beta[token * config.value_heads + value_head]);
    float result = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const float decayed = __fmul_rn(state[key_lane], decay);
      const float updated = __fadd_rn(
          decayed,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), delta));
      state[key_lane] = updated;
      result = __fadd_rn(
          result,
          __fmul_rn(__fmul_rn(query[key_lane], query_inverse), updated));
    }
    output[(token * config.value_heads + value_head) * config.value_width +
           lane] = result;
    __syncthreads();
  }
  for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
    const std::size_t index =
        head_base + static_cast<std::size_t>(key_lane) * config.value_width +
        lane;
    candidate[index] = state[key_lane];
  }
}

template <int kFixedWidth>
__global__ void prepare_recurrence_windows_zero(
    GdnConfig config, const float* convolution_output, const float* log_decay,
    const float* beta, float* scratch, std::size_t token_count,
    std::size_t batch_token_start, bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t key_width =
      kFixedWidth > 0 ? static_cast<std::uint32_t>(kFixedWidth)
                      : config.key_width;
  const std::uint32_t value_width =
      kFixedWidth > 0 ? static_cast<std::uint32_t>(kFixedWidth)
                      : config.value_width;
  if (value_head >= config.value_heads || lane >= value_width) return;
  const std::size_t local_window = static_cast<std::size_t>(blockIdx.y);
  const std::size_t batch_windows = static_cast<std::size_t>(gridDim.y);
  const std::size_t token_offset =
      batch_token_start + local_window * kScanWindow;
  if (token_offset >= token_count) return;
  const std::size_t remaining_tokens = token_count - token_offset;
  const std::size_t window_tokens =
      remaining_tokens < kScanWindow ? remaining_tokens : kScanWindow;
  const std::size_t recurrent_values =
      static_cast<std::size_t>(config.value_heads) * key_width * value_width;
  const std::size_t operator_values =
      static_cast<std::size_t>(config.value_heads) * key_width * key_width;
  float* candidate = scratch + local_window * recurrent_values;
  float* window_a = scratch + batch_windows * recurrent_values +
                    local_window * operator_values;
  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::uint32_t key_head = value_head / reuse;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * key_width;
  __shared__ float key_inverse;
  __shared__ float key_scale[kMaximumWidth];
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * key_width * value_width;
  const std::size_t a_head =
      static_cast<std::size_t>(value_head) * key_width * key_width;
  const std::size_t channels = 2 * query_count +
                               static_cast<std::size_t>(config.value_heads) *
                                   value_width;
  const bool owns_a = lane < key_width;
  constexpr int kColWidth = kFixedWidth > 0 ? kFixedWidth : kMaximumWidth;
  float a_col[kColWidth];
  if (owns_a) {
    for (std::uint32_t row = 0; row < key_width; ++row) {
      a_col[row] = row == lane ? 1.0F : 0.0F;
    }
  }
  for (std::size_t token = 0; token < window_tokens; ++token) {
    const float* token_convolution =
        convolution_output + (token_offset + token) * channels;
    const float* key = token_convolution + query_count + key_head * key_width;
    const std::uint32_t replica = value_head % reuse;
    const std::uint32_t tiled_head = replica * config.key_heads + key_head;
    const std::uint32_t source_head = value_is_tiled ? tiled_head : value_head;
    const float* value = token_convolution + 2 * query_count +
                         source_head * value_width;
    if (lane == 0) {
      float key_squares = 0.0F;
      for (std::uint32_t index = 0; index < key_width; ++index) {
        key_squares =
            __fadd_rn(key_squares, __fmul_rn(key[index], key[index]));
      }
      key_inverse = 1.0F / sqrtf(key_squares + kL2Epsilon);
    }
    __syncthreads();
    if (lane < key_width) {
      key_scale[lane] = __fmul_rn(key[lane], key_inverse);
    }
    __syncthreads();

    const float decay = expf(
        log_decay[(token_offset + token) * config.value_heads + value_head]);
    const float gate =
        beta[(token_offset + token) * config.value_heads + value_head];
    float prediction = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * value_width + lane;
      const float current = token == 0 ? 0.0F : candidate[index];
      const float decayed = __fmul_rn(current, decay);
      prediction = __fadd_rn(prediction, __fmul_rn(key_scale[key_lane], decayed));
    }
    const float delta = __fmul_rn(value[lane] - prediction, gate);
    for (std::uint32_t key_lane = 0; key_lane < key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * value_width + lane;
      const float current = token == 0 ? 0.0F : candidate[index];
      const float decayed = __fmul_rn(current, decay);
      candidate[index] =
          __fadd_rn(decayed, __fmul_rn(key_scale[key_lane], delta));
    }
    if (owns_a) {
      float a_prediction = 0.0F;
      for (std::uint32_t key_lane = 0; key_lane < key_width; ++key_lane) {
        const float decayed = __fmul_rn(a_col[key_lane], decay);
        a_prediction = __fadd_rn(
            a_prediction, __fmul_rn(key_scale[key_lane], decayed));
      }
      const float a_delta = __fmul_rn(0.0F - a_prediction, gate);
      for (std::uint32_t key_lane = 0; key_lane < key_width; ++key_lane) {
        const float decayed = __fmul_rn(a_col[key_lane], decay);
        a_col[key_lane] = __fadd_rn(
            decayed, __fmul_rn(key_scale[key_lane], a_delta));
      }
    }
    __syncthreads();
  }
  if (owns_a) {
    for (std::uint32_t row = 0; row < key_width; ++row) {
      window_a[a_head + static_cast<std::size_t>(row) * key_width + lane] =
          a_col[row];
    }
  }
}

__global__ void __launch_bounds__(128, 1) scan_window_prefix_states_w128(
    GdnConfig config, const float* source, float* candidate, float* scratch,
    std::size_t batch_windows) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  constexpr std::uint32_t kW = 128;
  if (value_head >= config.value_heads || lane >= kW) return;
  const std::size_t recurrent_values =
      static_cast<std::size_t>(config.value_heads) * kW * kW;
  const std::size_t operator_values = recurrent_values;
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * kW * kW;
  const std::size_t a_head = head_base;
  float s[kW];
#pragma unroll
  for (std::uint32_t key_lane = 0; key_lane < kW; ++key_lane) {
    s[key_lane] = source[head_base + static_cast<std::size_t>(key_lane) * kW +
                         lane];
  }
  for (std::size_t local = 0; local < batch_windows; ++local) {
    float* window_b = scratch + local * recurrent_values;
    const float* window_a = scratch + batch_windows * recurrent_values +
                            local * operator_values;
    for (std::uint32_t row = 0; row < kW; ++row) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(row) * kW + lane;
      const float b = window_b[index];
      float acc = 0.0F;
      const float* a_row =
          window_a + a_head + static_cast<std::size_t>(row) * kW;
#pragma unroll
      for (std::uint32_t k_in = 0; k_in < kW; ++k_in) {
        acc = __fadd_rn(acc, __fmul_rn(a_row[k_in], s[k_in]));
      }
      candidate[index] = __fadd_rn(acc, b);
    }
#pragma unroll
    for (std::uint32_t key_lane = 0; key_lane < kW; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * kW + lane;
      window_b[index] = s[key_lane];
      s[key_lane] = candidate[index];
    }
  }
}

template <int kFixedWidth>
__global__ void scan_window_prefix_states(
    GdnConfig config, const float* source, float* candidate, float* scratch,
    std::size_t batch_windows) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t key_width =
      kFixedWidth > 0 ? static_cast<std::uint32_t>(kFixedWidth)
                      : config.key_width;
  const std::uint32_t value_width =
      kFixedWidth > 0 ? static_cast<std::uint32_t>(kFixedWidth)
                      : config.value_width;
  if (value_head >= config.value_heads || lane >= value_width) return;
  const std::size_t recurrent_values =
      static_cast<std::size_t>(config.value_heads) * key_width * value_width;
  const std::size_t operator_values =
      static_cast<std::size_t>(config.value_heads) * key_width * key_width;
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * key_width * value_width;
  const std::size_t a_head =
      static_cast<std::size_t>(value_head) * key_width * key_width;
  constexpr int kColWidth = kFixedWidth > 0 ? kFixedWidth : kMaximumWidth;
  float s[kColWidth];
#pragma unroll
  for (std::uint32_t key_lane = 0; key_lane < kColWidth; ++key_lane) {
    if (kFixedWidth > 0 || key_lane < key_width) {
      s[key_lane] = source[head_base +
                           static_cast<std::size_t>(key_lane) * value_width +
                           lane];
    }
  }
  for (std::size_t local = 0; local < batch_windows; ++local) {
    float* window_b = scratch + local * recurrent_values;
    const float* window_a = scratch + batch_windows * recurrent_values +
                            local * operator_values;
    float saved_b[kColWidth];
#pragma unroll
    for (std::uint32_t key_lane = 0; key_lane < kColWidth; ++key_lane) {
      if (kFixedWidth > 0 || key_lane < key_width) {
        const std::size_t index =
            head_base + static_cast<std::size_t>(key_lane) * value_width +
            lane;
        saved_b[key_lane] = window_b[index];
        window_b[index] = s[key_lane];
      }
    }
    float s_next[kColWidth];
    for (std::uint32_t row = 0; row < key_width; ++row) {
      float acc = 0.0F;
      const float* a_row =
          window_a + a_head + static_cast<std::size_t>(row) * key_width;
#pragma unroll
      for (std::uint32_t k_in = 0; k_in < kColWidth; ++k_in) {
        if (kFixedWidth > 0 || k_in < key_width) {
          acc = __fadd_rn(acc, __fmul_rn(a_row[k_in], s[k_in]));
        }
      }
      s_next[row] = __fadd_rn(acc, saved_b[row]);
    }
#pragma unroll
    for (std::uint32_t key_lane = 0; key_lane < kColWidth; ++key_lane) {
      if (kFixedWidth > 0 || key_lane < key_width) {
        s[key_lane] = s_next[key_lane];
      }
    }
  }
  for (std::uint32_t key_lane = 0; key_lane < key_width; ++key_lane) {
    candidate[head_base + static_cast<std::size_t>(key_lane) * value_width +
              lane] = s[key_lane];
  }
}

__global__ void prepare_recurrence_windows_from_state(
    GdnConfig config, const float* convolution_output, const float* log_decay,
    const float* beta, float* scratch, float* candidate, float* output,
    std::size_t token_count, std::size_t batch_token_start,
    bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (value_head >= config.value_heads || lane >= config.value_width) return;
  const std::size_t local_window = static_cast<std::size_t>(blockIdx.y);
  const std::size_t batch_windows = static_cast<std::size_t>(gridDim.y);
  const std::size_t token_offset =
      batch_token_start + local_window * kScanWindow;
  if (token_offset >= token_count) return;
  const std::size_t remaining_tokens = token_count - token_offset;
  const std::size_t window_tokens =
      remaining_tokens < kScanWindow ? remaining_tokens : kScanWindow;
  const std::size_t recurrent_values =
      static_cast<std::size_t>(config.value_heads) * config.key_width *
      config.value_width;
  float* window_state = scratch + local_window * recurrent_values;
  const bool last_in_batch = local_window + 1 == batch_windows;
  const float* source = window_state;
  float* dest = window_state;
  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::uint32_t key_head = value_head / reuse;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  __shared__ float query_inverse;
  __shared__ float key_inverse;
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * config.key_width *
      config.value_width;
  const std::size_t channels = 2 * query_count +
                               static_cast<std::size_t>(config.value_heads) *
                                   config.value_width;
  for (std::size_t token = 0; token < window_tokens; ++token) {
    const float* token_convolution =
        convolution_output + (token_offset + token) * channels;
    const float* query = token_convolution + key_head * config.key_width;
    const float* key = token_convolution + query_count +
                       key_head * config.key_width;
    const std::uint32_t replica = value_head % reuse;
    const std::uint32_t tiled_head = replica * config.key_heads + key_head;
    const std::uint32_t source_head = value_is_tiled ? tiled_head : value_head;
    const float* value = token_convolution + 2 * query_count +
                         source_head * config.value_width;
    if (lane == 0) {
      float query_squares = 0.0F;
      float key_squares = 0.0F;
      for (std::uint32_t index = 0; index < config.key_width; ++index) {
        query_squares = __fadd_rn(
            query_squares, __fmul_rn(query[index], query[index]));
        key_squares =
            __fadd_rn(key_squares, __fmul_rn(key[index], key[index]));
      }
      query_inverse =
          1.0F / sqrtf(query_squares + kL2Epsilon) /
          sqrtf(static_cast<float>(config.key_width));
      key_inverse = 1.0F / sqrtf(key_squares + kL2Epsilon);
    }
    __syncthreads();

    const float* current = token == 0 ? source : dest;
    const float decay = expf(
        log_decay[(token_offset + token) * config.value_heads + value_head]);
    float prediction = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          lane;
      const float decayed = __fmul_rn(current[index], decay);
      prediction = __fadd_rn(
          prediction,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), decayed));
    }
    const float delta = __fmul_rn(
        value[lane] - prediction,
        beta[(token_offset + token) * config.value_heads + value_head]);
    float result = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          lane;
      const float decayed = __fmul_rn(current[index], decay);
      const float updated = __fadd_rn(
          decayed,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), delta));
      dest[index] = updated;
      result = __fadd_rn(
          result,
          __fmul_rn(__fmul_rn(query[key_lane], query_inverse), updated));
    }
    output[((token_offset + token) * config.value_heads + value_head) *
               config.value_width +
           lane] = result;
    __syncthreads();
  }
  if (last_in_batch) {
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          lane;
      candidate[index] = window_state[index];
    }
  }
}

__global__ void commit_state(const float* candidate_convolution,
                             float* committed_convolution,
                             std::size_t convolution_values,
                             const float* candidate_recurrent,
                             float* committed_recurrent,
                             std::size_t recurrent_values,
                             std::uint64_t new_frontier,
                             std::uint64_t* committed_frontier) {
  for (std::size_t index = threadIdx.x; index < convolution_values;
       index += blockDim.x) {
    committed_convolution[index] = candidate_convolution[index];
  }
  for (std::size_t index = threadIdx.x; index < recurrent_values;
       index += blockDim.x) {
    committed_recurrent[index] = candidate_recurrent[index];
  }
  __syncthreads();
  if (threadIdx.x == 0) *committed_frontier = new_frontier;
}

}  // namespace

namespace {

cudaError_t launch_gdn_prepare_chunk_layout(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream, bool value_is_tiled,
    GdnScanPath path, float* scratch, std::size_t scratch_floats) noexcept;

}  // namespace

std::size_t gdn_convolution_channels(const GdnConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return 2 * static_cast<std::size_t>(config.key_heads) * config.key_width +
         static_cast<std::size_t>(config.value_heads) * config.value_width;
}

std::size_t gdn_convolution_values(const GdnConfig& config) noexcept {
  return gdn_convolution_channels(config) * config.convolution_width;
}

std::size_t gdn_recurrent_values(const GdnConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return static_cast<std::size_t>(config.value_heads) * config.key_width *
         config.value_width;
}

std::size_t gdn_output_values(const GdnConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return static_cast<std::size_t>(config.value_heads) * config.value_width;
}

std::size_t gdn_scan_window_count(std::size_t token_count) noexcept {
  if (token_count == 0) return 0;
  return (token_count + kScanWindow - 1) / kScanWindow;
}

std::size_t gdn_scan_operator_values(const GdnConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return static_cast<std::size_t>(config.value_heads) * config.key_width *
         config.key_width;
}

std::size_t gdn_scan_scratch_floats(const GdnConfig& config,
                                    std::size_t token_count) noexcept {
  const std::size_t recurrent = gdn_recurrent_values(config);
  const std::size_t operators = gdn_scan_operator_values(config);
  if (recurrent == 0 || operators == 0) return 0;
  return gdn_scan_window_count(token_count) * (recurrent + operators);
}

cudaError_t launch_gdn_prepare(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, const GdnState& committed, const GdnState& candidate,
    float* convolution_output, float* recurrent_output,
    cudaStream_t stream) noexcept {
  return launch_gdn_prepare_chunk(
      config, convolution_input, convolution_weights, log_decay, beta, 1,
      committed, candidate, convolution_output, recurrent_output, stream);
}

cudaError_t launch_gdn_prepare_tiled(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, const GdnState& committed, const GdnState& candidate,
    float* convolution_output, float* recurrent_output,
    cudaStream_t stream) noexcept {
  return launch_gdn_prepare_chunk_layout(
      config, convolution_input, convolution_weights, log_decay, beta, 1,
      committed, candidate, convolution_output, recurrent_output, stream, true,
      GdnScanPath::kSequentialWindows, nullptr, 0);
}

cudaError_t launch_gdn_prepare_chunk(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream, GdnScanPath path,
    float* scratch, std::size_t scratch_floats) noexcept {
  return launch_gdn_prepare_chunk_layout(
      config, convolution_input, convolution_weights, log_decay, beta,
      token_count, committed, candidate, convolution_output, recurrent_output,
      stream, false, path, scratch, scratch_floats);
}

cudaError_t launch_gdn_prepare_chunk_tiled(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream, GdnScanPath path,
    float* scratch, std::size_t scratch_floats) noexcept {
  return launch_gdn_prepare_chunk_layout(
      config, convolution_input, convolution_weights, log_decay, beta,
      token_count, committed, candidate, convolution_output, recurrent_output,
      stream, true, path, scratch, scratch_floats);
}

namespace {

cudaError_t launch_sequential_windows(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream,
    bool value_is_tiled) noexcept {
  const std::size_t channels = gdn_convolution_channels(config);
  const unsigned int blocks =
      static_cast<unsigned int>((channels + kThreads - 1) / kThreads);
  for (std::size_t start = 0; start < token_count; start += kScanWindow) {
    const std::size_t window = std::min(kScanWindow, token_count - start);
    const float* source_convolution =
        start == 0 ? committed.convolution : candidate.convolution;
    const float* source_recurrent =
        start == 0 ? committed.recurrent : candidate.recurrent;
    prepare_convolution_window<<<blocks, kThreads, 0, stream>>>(
        convolution_input + start * channels, convolution_weights,
        source_convolution, candidate.convolution,
        convolution_output + start * channels, channels,
        config.convolution_width, window);
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (window == 1 && gdn_decode_uses_persistent_transposed()) {
      error = launch_gdn_decode_persistent_transposed_recurrence(
          config, convolution_output + start * channels,
          log_decay + start * config.value_heads,
          beta + start * config.value_heads, source_recurrent,
          candidate.recurrent,
          recurrent_output + start * gdn_output_values(config), value_is_tiled,
          stream);
    } else if (window == 1 && gdn_decode_uses_transposed()) {
      error = launch_gdn_decode_transposed_recurrence(
          config, convolution_output + start * channels,
          log_decay + start * config.value_heads,
          beta + start * config.value_heads, source_recurrent,
          candidate.recurrent,
          recurrent_output + start * gdn_output_values(config), value_is_tiled,
          stream);
    } else if (window == 1 && gdn_decode_uses_tiled()) {
      error = launch_gdn_decode_tiled_recurrence(
          config, convolution_output + start * channels,
          log_decay + start * config.value_heads,
          beta + start * config.value_heads, source_recurrent,
          candidate.recurrent,
          recurrent_output + start * gdn_output_values(config), value_is_tiled,
          stream);
    } else {
      prepare_recurrence_window<<<config.value_heads, kThreads, 0, stream>>>(
          config, convolution_output + start * channels,
          log_decay + start * config.value_heads,
          beta + start * config.value_heads, source_recurrent,
          candidate.recurrent,
          recurrent_output + start * gdn_output_values(config), window,
          value_is_tiled);
      error = cudaPeekAtLastError();
      if (error == cudaSuccess) {
        record_gdn_decode_launch(kGdnDecodeLaunchVariantSequential, 0,
                                 config.value_heads, 1);
      }
    }
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

cudaError_t launch_convolution_then_recurrence(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream, bool value_is_tiled,
    bool quality) noexcept {
  const std::size_t channels = gdn_convolution_channels(config);
  const dim3 conv_grid(
      static_cast<unsigned int>((channels + kConvolutionThreads - 1) /
                                kConvolutionThreads),
      static_cast<unsigned int>(token_count));
  cudaError_t error = quartz_launch_kernel(
      prepare_convolution_chunk_parallel, conv_grid, dim3(kConvolutionThreads),
      0, stream, convolution_input, convolution_weights, committed.convolution,
      candidate.convolution, convolution_output, channels,
      config.convolution_width, token_count);
  if (error != cudaSuccess) return error;
  if (quality && config.key_width == 128 && config.value_width == 128) {
    return launch_gdn_quality_recurrence(
        config, convolution_output, log_decay, beta, committed.recurrent,
        candidate.recurrent, recurrent_output, token_count, value_is_tiled,
        stream);
  }
  prepare_recurrence_fused_token_loop<<<config.value_heads, kThreads, 0,
                                        stream>>>(
      config, convolution_output, log_decay, beta, committed.recurrent,
      candidate.recurrent, recurrent_output, token_count, value_is_tiled);
  return cudaPeekAtLastError();
}

cudaError_t launch_fused_token_loop(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream, bool value_is_tiled,
    float* scratch, std::size_t scratch_floats) noexcept {
  return launch_gdn_quality_fused(
      config, convolution_input, convolution_weights, log_decay, beta,
      token_count, committed, candidate, convolution_output, recurrent_output,
      nullptr, nullptr, nullptr, stream, value_is_tiled, nullptr, nullptr,
      scratch, scratch_floats);
}

cudaError_t launch_parallel_associative(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream, bool value_is_tiled,
    float* scratch, std::size_t scratch_floats) noexcept {
  const std::size_t channels = gdn_convolution_channels(config);
  const std::size_t recurrent_values = gdn_recurrent_values(config);
  const dim3 conv_grid(
      static_cast<unsigned int>((channels + kConvolutionThreads - 1) /
                                kConvolutionThreads),
      static_cast<unsigned int>(token_count));
  cudaError_t error = quartz_launch_kernel(
      prepare_convolution_chunk_parallel, conv_grid, dim3(kConvolutionThreads),
      0, stream, convolution_input, convolution_weights, committed.convolution,
      candidate.convolution, convolution_output, channels,
      config.convolution_width, token_count);
  if (error != cudaSuccess) return error;
  const std::size_t num_windows = gdn_scan_window_count(token_count);
  if (num_windows < 2) {
    prepare_recurrence_window<<<config.value_heads, kThreads, 0, stream>>>(
        config, convolution_output, log_decay, beta, committed.recurrent,
        candidate.recurrent, recurrent_output, token_count, value_is_tiled);
    return cudaPeekAtLastError();
  }
  const std::size_t operator_values = gdn_scan_operator_values(config);
  const std::size_t pair = recurrent_values + operator_values;
  const std::size_t fit = scratch_floats / pair;
  for (std::size_t window_start = 0; window_start < num_windows;) {
    const std::size_t remaining = num_windows - window_start;
    const std::size_t batch = std::min(fit, remaining);
    const std::size_t token_start = window_start * kScanWindow;
    const dim3 intra_grid(config.value_heads,
                           static_cast<unsigned int>(batch));
    if (config.key_width == 128 && config.value_width == 128) {
      prepare_recurrence_windows_zero<128>
          <<<intra_grid, kThreads, 0, stream>>>(
              config, convolution_output, log_decay, beta, scratch, token_count,
              token_start, value_is_tiled);
    } else {
      prepare_recurrence_windows_zero<0><<<intra_grid, kThreads, 0, stream>>>(
          config, convolution_output, log_decay, beta, scratch, token_count,
          token_start, value_is_tiled);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    const float* source =
        window_start == 0 ? committed.recurrent : candidate.recurrent;
    if (config.key_width == 128 && config.value_width == 128) {
      scan_window_prefix_states_w128<<<config.value_heads, kThreads, 0, stream>>>(
          config, source, candidate.recurrent, scratch, batch);
    } else {
      scan_window_prefix_states<0><<<config.value_heads, kThreads, 0, stream>>>(
          config, source, candidate.recurrent, scratch, batch);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    prepare_recurrence_windows_from_state<<<intra_grid, kThreads, 0, stream>>>(
        config, convolution_output, log_decay, beta, scratch,
        candidate.recurrent, recurrent_output, token_count, token_start,
        value_is_tiled);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    window_start += batch;
  }
  return cudaSuccess;
}

cudaError_t launch_gdn_prepare_chunk_layout(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream, bool value_is_tiled,
    GdnScanPath path, float* scratch, std::size_t scratch_floats) noexcept {
  const std::size_t channels = gdn_convolution_channels(config);
  const bool allow_null_conv =
      path == GdnScanPath::kFusedTokenLoop && gdn_fuses_conv();
  if (channels == 0 || token_count == 0 || convolution_input == nullptr ||
      convolution_weights == nullptr || log_decay == nullptr ||
      beta == nullptr || committed.convolution == nullptr ||
      committed.recurrent == nullptr || candidate.convolution == nullptr ||
      candidate.recurrent == nullptr ||
      (convolution_output == nullptr && !allow_null_conv) ||
      recurrent_output == nullptr ||
      committed.convolution == candidate.convolution ||
      committed.recurrent == candidate.recurrent) {
    return cudaErrorInvalidValue;
  }
  if (path != GdnScanPath::kSequentialWindows &&
      path != GdnScanPath::kParallelAssociative &&
      path != GdnScanPath::kFusedTokenLoop) {
    return cudaErrorInvalidValue;
  }
  if (path == GdnScanPath::kFusedTokenLoop) {
    return launch_fused_token_loop(
        config, convolution_input, convolution_weights, log_decay, beta,
        token_count, committed, candidate, convolution_output,
        recurrent_output, stream, value_is_tiled, scratch, scratch_floats);
  }
  if (path == GdnScanPath::kParallelAssociative) {
    if (scratch == nullptr ||
        scratch_floats < gdn_recurrent_values(config) +
                             gdn_scan_operator_values(config)) {
      return cudaErrorInvalidValue;
    }
    return launch_parallel_associative(
        config, convolution_input, convolution_weights, log_decay, beta,
        token_count, committed, candidate, convolution_output, recurrent_output,
        stream, value_is_tiled, scratch, scratch_floats);
  }
  return launch_sequential_windows(
      config, convolution_input, convolution_weights, log_decay, beta,
      token_count, committed, candidate, convolution_output, recurrent_output,
      stream, value_is_tiled);
}

}  // namespace

cudaError_t launch_gdn_fused_rank2(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream,
    bool value_is_tiled) noexcept {
  const std::size_t channels = gdn_convolution_channels(config);
  if (channels == 0 || token_count == 0 || convolution_input == nullptr ||
      convolution_weights == nullptr || log_decay == nullptr ||
      beta == nullptr || committed.convolution == nullptr ||
      committed.recurrent == nullptr || candidate.convolution == nullptr ||
      candidate.recurrent == nullptr || convolution_output == nullptr ||
      recurrent_output == nullptr ||
      committed.convolution == candidate.convolution ||
      committed.recurrent == candidate.recurrent) {
    return cudaErrorInvalidValue;
  }
  return launch_convolution_then_recurrence(
      config, convolution_input, convolution_weights, log_decay, beta,
      token_count, committed, candidate, convolution_output, recurrent_output,
      stream, value_is_tiled, false);
}

cudaError_t launch_gdn_quality_fused(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, const float* gate_tiled, const float* norm,
    __nv_bfloat16* output_bf16, cudaStream_t stream, bool value_is_tiled,
    const char* path, const char* inverse_path, float* inverse_scratch,
    std::size_t inverse_scratch_floats, const char* preproc_path) noexcept {
  const std::size_t channels = gdn_convolution_channels(config);
  if (channels == 0 || token_count == 0 || log_decay == nullptr ||
      beta == nullptr || committed.recurrent == nullptr ||
      candidate.recurrent == nullptr || recurrent_output == nullptr ||
      committed.recurrent == candidate.recurrent) {
    return cudaErrorInvalidValue;
  }
  const char* selected = gdn_fuse_path_or_selected(path);
  bool fuse_gate = gdn_path_eq(selected, "fuse_gate") ||
                   gdn_path_eq(selected, "fuse_both");
  const bool want_conv = gdn_path_eq(selected, "fuse_conv") ||
                         gdn_path_eq(selected, "fuse_both");
  if (want_conv && (convolution_input == nullptr ||
                    convolution_weights == nullptr ||
                    committed.convolution == nullptr ||
                    candidate.convolution == nullptr)) {
    selected = "off";
    fuse_gate = false;
  }
  if (fuse_gate && (gate_tiled == nullptr || norm == nullptr ||
                    output_bf16 == nullptr)) {
    selected = "off";
    fuse_gate = false;
  }
  if (config.key_width != 128 || config.value_width != 128 ||
      config.convolution_width != 4) {
    selected = "off";
    fuse_gate = false;
  }
  const bool launch_split_gate =
      !fuse_gate && gate_tiled != nullptr && norm != nullptr &&
      output_bf16 != nullptr;
  auto launch_parallel_conv = [&]() -> cudaError_t {
    if (convolution_input == nullptr || convolution_weights == nullptr ||
        committed.convolution == nullptr || candidate.convolution == nullptr ||
        convolution_output == nullptr) {
      return cudaErrorInvalidValue;
    }
    const dim3 conv_grid(
        static_cast<unsigned int>((channels + kConvolutionThreads - 1) /
                                  kConvolutionThreads),
        static_cast<unsigned int>(token_count));
    return quartz_launch_kernel(
        prepare_convolution_chunk_parallel, conv_grid,
        dim3(kConvolutionThreads), 0, stream, convolution_input,
        convolution_weights, committed.convolution, candidate.convolution,
        convolution_output, channels, config.convolution_width, token_count);
  };
  cudaError_t error = cudaSuccess;
  if (gdn_path_eq(selected, "fuse_both")) {
    prepare_recurrence_fused_head<true>
        <<<config.value_heads, dim3(32U, 32U, 1U), 0, stream>>>(
            config, convolution_input, convolution_weights,
            committed.convolution, candidate.convolution, convolution_output,
            log_decay, beta, committed.recurrent, candidate.recurrent,
            recurrent_output, gate_tiled, norm, output_bf16, token_count,
            value_is_tiled);
    error = cudaPeekAtLastError();
  } else if (gdn_path_eq(selected, "fuse_gate")) {
    error = launch_parallel_conv();
    if (error == cudaSuccess) {
      prepare_recurrence_fused_head<false>
          <<<config.value_heads, dim3(32U, 32U, 1U), 0, stream>>>(
              config, convolution_input, convolution_weights,
              committed.convolution, candidate.convolution, convolution_output,
              log_decay, beta, committed.recurrent, candidate.recurrent,
              recurrent_output, gate_tiled, norm, output_bf16, token_count,
              value_is_tiled);
      error = cudaPeekAtLastError();
    }
  } else if (gdn_path_eq(selected, "fuse_conv")) {
    const dim3 grid(config.value_heads, 1U, config.value_width / 4U);
    const dim3 block(32U, 4U, 1U);
    prepare_recurrence_fused_conv<<<grid, block, 0, stream>>>(
        config, convolution_input, convolution_weights, committed.convolution,
        candidate.convolution, convolution_output, log_decay, beta,
        committed.recurrent, candidate.recurrent, recurrent_output, token_count,
        value_is_tiled);
    error = cudaPeekAtLastError();
  } else if (gdn_can_launch_preproc(config, token_count, selected, preproc_path,
                                   inverse_scratch, inverse_scratch_floats)) {
    const char* preproc = gdn_preproc_path_or_selected(preproc_path);
    error = launch_parallel_conv();
    if (error == cudaSuccess) {
      error = launch_gdn_scaled_qk(config, convolution_output, token_count,
                                   inverse_scratch, stream);
    }
    if (error == cudaSuccess) {
      error = launch_gdn_decay(
          config, log_decay, token_count,
          gdn_preproc_decay_ptr(inverse_scratch, token_count, config.key_heads,
                                config.key_width),
          gdn_preproc_uses_fast_exp(preproc), stream);
    }
    const bool transposed = gdn_preproc_uses_transpose(preproc);
    const float* recurrence_source = committed.recurrent;
    float* recurrence_candidate = candidate.recurrent;
    if (error == cudaSuccess && transposed) {
      float* state_t =
          gdn_preproc_state_t_ptr(inverse_scratch, config, token_count);
      error = launch_gdn_transpose_state_in(config, committed.recurrent, state_t,
                                            stream);
      recurrence_source = state_t;
      recurrence_candidate = state_t;
    }
    if (error == cudaSuccess) {
      error = launch_gdn_quality_recurrence_preproc(
          config, convolution_output,
          gdn_preproc_decay_ptr(inverse_scratch, token_count, config.key_heads,
                                config.key_width),
          beta, recurrence_source, recurrence_candidate, recurrent_output,
          inverse_scratch, token_count, value_is_tiled,
          gdn_preproc_uses_fma(preproc), transposed, stream);
    }
    if (error == cudaSuccess && transposed) {
      error = launch_gdn_transpose_state_out(
          config, recurrence_candidate, candidate.recurrent, stream);
    }
  } else if (gdn_can_launch_shared_inverse(config, token_count, selected,
                                          inverse_path, inverse_scratch,
                                          inverse_scratch_floats)) {
    error = launch_parallel_conv();
    if (error == cudaSuccess) {
      error = launch_gdn_shared_inverses(config, convolution_output,
                                         token_count, inverse_scratch, stream);
    }
    if (error == cudaSuccess) {
      error = launch_gdn_quality_recurrence_shared(
          config, convolution_output, log_decay, beta, committed.recurrent,
          candidate.recurrent, recurrent_output, inverse_scratch, token_count,
          value_is_tiled, stream);
    }
  } else {
    error = launch_convolution_then_recurrence(
        config, convolution_input, convolution_weights, log_decay, beta,
        token_count, committed, candidate, convolution_output, recurrent_output,
        stream, value_is_tiled, true);
  }
  if (error == cudaSuccess && launch_split_gate) {
    error = launch_gdn_gated_output_rows(
        recurrent_output, gate_tiled, norm, config.key_heads,
        config.value_heads / config.key_heads, config.value_width, token_count,
        output_bf16, stream);
  }
  return error;
}

int gdn_fused_quality_occupancy() noexcept {
  return gdn_fuse_occupancy("off");
}

int gdn_decode_transposed_occupancy() noexcept {
  int occupancy = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, prepare_recurrence_decode_transposed, kGdnDecodeThreads, 0);
  return occupancy;
}

void gdn_decode_transposed_attributes(int* registers, std::size_t* local_bytes,
                                      int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  cudaFuncGetAttributes(&attrs, prepare_recurrence_decode_transposed);
  if (registers != nullptr) *registers = attrs.numRegs;
  if (local_bytes != nullptr) {
    *local_bytes = static_cast<std::size_t>(attrs.localSizeBytes);
  }
  if (occupancy != nullptr) {
    *occupancy = gdn_decode_transposed_occupancy();
  }
}

int gdn_decode_tiled_occupancy(unsigned int value_tile) noexcept {
  int occupancy = 0;
  if (value_tile == kGdnDecodeValueTile16) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, prepare_recurrence_decode_tiled<kGdnDecodeValueTile16>,
        kGdnDecodeThreads, 0);
  } else if (value_tile == kGdnDecodeValueTile32) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, prepare_recurrence_decode_tiled<kGdnDecodeValueTile32>,
        kGdnDecodeThreads, 0);
  }
  return occupancy;
}

void gdn_decode_tiled_attributes(unsigned int value_tile, int* registers,
                                 std::size_t* local_bytes,
                                 int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  if (value_tile == kGdnDecodeValueTile16) {
    cudaFuncGetAttributes(
        &attrs, prepare_recurrence_decode_tiled<kGdnDecodeValueTile16>);
  } else if (value_tile == kGdnDecodeValueTile32) {
    cudaFuncGetAttributes(
        &attrs, prepare_recurrence_decode_tiled<kGdnDecodeValueTile32>);
  }
  if (registers != nullptr) *registers = attrs.numRegs;
  if (local_bytes != nullptr) {
    *local_bytes = static_cast<std::size_t>(attrs.localSizeBytes);
  }
  if (occupancy != nullptr) {
    *occupancy = gdn_decode_tiled_occupancy(value_tile);
  }
}

cudaError_t launch_gdn_recurrence_only(
    const GdnConfig& config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, bool value_is_tiled,
    cudaStream_t stream) noexcept {
  if (convolution_output == nullptr || log_decay == nullptr ||
      beta == nullptr || source == nullptr || candidate == nullptr ||
      output == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (gdn_decode_uses_persistent_transposed()) {
    return launch_gdn_decode_persistent_transposed_recurrence(
        config, convolution_output, log_decay, beta, source, candidate, output,
        value_is_tiled, stream);
  }
  if (gdn_decode_uses_transposed()) {
    return launch_gdn_decode_transposed_recurrence(
        config, convolution_output, log_decay, beta, source, candidate, output,
        value_is_tiled, stream);
  }
  if (gdn_decode_uses_tiled()) {
    return launch_gdn_decode_tiled_recurrence(
        config, convolution_output, log_decay, beta, source, candidate, output,
        value_is_tiled, stream);
  }
  prepare_recurrence_window<<<config.value_heads, kThreads, 0, stream>>>(
      config, convolution_output, log_decay, beta, source, candidate, output, 1,
      value_is_tiled);
  cudaError_t error = cudaPeekAtLastError();
  if (error == cudaSuccess) {
    record_gdn_decode_launch(kGdnDecodeLaunchVariantSequential, 0,
                             config.value_heads, 1);
  }
  return error;
}

cudaError_t launch_gdn_convert_recurrent_layout(
    const GdnConfig& config, float* recurrent, bool to_col_major,
    cudaStream_t stream, GdnConversionBoundary boundary) noexcept {
  if (recurrent == nullptr || config.value_heads == 0 ||
      config.value_width == 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, kGdnDecodeWarpsPerCta);
  const std::size_t smem =
      static_cast<std::size_t>(config.value_width) * config.value_width *
      sizeof(float);
  cudaError_t error = gdn_set_relayout_dynamic_smem(smem);
  if (error != cudaSuccess) return error;
  if (to_col_major) {
    gdn_relayout_row_to_col_inplace<<<config.value_heads, block, smem, stream>>>(
        recurrent, config.value_heads, config.value_width);
  } else {
    gdn_relayout_col_to_row_inplace<<<config.value_heads, block, smem, stream>>>(
        recurrent, config.value_heads, config.value_width);
  }
  error = cudaPeekAtLastError();
  if (error == cudaSuccess) {
    const std::size_t bytes = static_cast<std::size_t>(config.value_heads) *
                              config.key_width * config.value_width *
                              sizeof(float);
    gdn_record_conversion(boundary, 1, bytes);
  }
  return error;
}

cudaError_t launch_gdn_copy_converted_recurrent(
    const GdnConfig& config, const float* source, float* dest, bool to_col_major,
    cudaStream_t stream, GdnConversionBoundary boundary) noexcept {
  if (source == nullptr || dest == nullptr) return cudaErrorInvalidValue;
  cudaError_t error =
      to_col_major ? launch_gdn_transpose_state_in(config, source, dest, stream)
                   : launch_gdn_transpose_state_out(config, source, dest,
                                                    stream);
  if (error == cudaSuccess) {
    const std::size_t bytes = static_cast<std::size_t>(config.value_heads) *
                              config.key_width * config.value_width *
                              sizeof(float);
    gdn_record_conversion(boundary, 1, bytes);
  }
  return error;
}

cudaError_t launch_gdn_convert_session_recurrent(
    float* recurrent, std::size_t layers, bool to_col_major,
    cudaStream_t stream, GdnConversionBoundary boundary) noexcept {
  if (recurrent == nullptr || layers == 0) return cudaErrorInvalidValue;
  const GdnConfig config{16, 48, 128, 128, 4};
  const std::size_t layer_floats =
      static_cast<std::size_t>(config.value_heads) * config.key_width *
      config.value_width;
  cudaError_t error = cudaSuccess;
  for (std::size_t layer = 0; layer < layers && error == cudaSuccess; ++layer) {
    error = launch_gdn_convert_recurrent_layout(
        config, recurrent + layer * layer_floats, to_col_major, stream,
        boundary);
  }
  return error;
}

cudaError_t launch_gdn_commit(const GdnConfig& config,
                              const GdnState& candidate,
                              const GdnState& committed,
                              std::uint64_t new_frontier,
                              std::uint64_t* committed_frontier,
                              cudaStream_t stream) noexcept {
  const std::size_t convolution_values = gdn_convolution_values(config);
  const std::size_t recurrent_values = gdn_recurrent_values(config);
  if (convolution_values == 0 || recurrent_values == 0 ||
      candidate.convolution == nullptr || candidate.recurrent == nullptr ||
      committed.convolution == nullptr || committed.recurrent == nullptr ||
      committed_frontier == nullptr ||
      candidate.convolution == committed.convolution ||
      candidate.recurrent == committed.recurrent) {
    return cudaErrorInvalidValue;
  }
  commit_state<<<1, 256, 0, stream>>>(
      candidate.convolution, committed.convolution, convolution_values,
      candidate.recurrent, committed.recurrent, recurrent_values, new_frontier,
      committed_frontier);
  return cudaPeekAtLastError();
}

}  // namespace qw38::cuda
