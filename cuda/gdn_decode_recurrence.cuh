#pragma once

// OPT-077 one-token parallel GDN recurrence. Included from gdn_step.cu inside
// namespace qw38::cuda. Row-major FP32 state is retained; llama.cpp
// gated_delta_net_cuda warp-sharding motivates the 2D tile, but its
// transposed S is not a drop-in. Convolution stays a separate launch.

constexpr float kDecodeGdnL2Epsilon = 1.0e-6F;

template <int kValueTile>
__global__ void __launch_bounds__(kGdnDecodeThreads, 2)
prepare_recurrence_decode_tiled(GdnConfig config,
                                const float* convolution_output,
                                const float* log_decay, const float* beta,
                                const float* source, float* candidate,
                                float* output, bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const int value_tile = kValueTile;
  const std::uint32_t v_col =
      blockIdx.y * static_cast<std::uint32_t>(value_tile) +
      (threadIdx.x % static_cast<std::uint32_t>(value_tile));
  const int k_tid = static_cast<int>(threadIdx.x / value_tile);
  const int key_threads = kGdnDecodeThreads / value_tile;
  if (value_head >= config.value_heads) return;

  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::uint32_t key_head = value_head / reuse;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  const std::size_t head_base = static_cast<std::size_t>(value_head) *
                                config.key_width * config.value_width;
  const std::uint32_t replica = value_head % reuse;
  const std::uint32_t tiled_head = replica * config.key_heads + key_head;
  const std::uint32_t source_head = value_is_tiled ? tiled_head : value_head;
  const float* query = convolution_output + key_head * config.key_width;
  const float* key =
      convolution_output + query_count + key_head * config.key_width;
  const float* value = convolution_output + 2 * query_count +
                       source_head * config.value_width;

  __shared__ float query_inverse;
  __shared__ float key_inverse;
  __shared__ float reduce_buf[kGdnDecodeThreads];
  if (threadIdx.x == 0) {
    float query_squares = 0.0F;
    float key_squares = 0.0F;
    for (std::uint32_t index = 0; index < config.key_width; ++index) {
      query_squares = __fadd_rn(query_squares,
                                __fmul_rn(query[index], query[index]));
      key_squares =
          __fadd_rn(key_squares, __fmul_rn(key[index], key[index]));
    }
    query_inverse = 1.0F / sqrtf(query_squares + kDecodeGdnL2Epsilon) /
                    sqrtf(static_cast<float>(config.key_width));
    key_inverse = 1.0F / sqrtf(key_squares + kDecodeGdnL2Epsilon);
  }
  __syncthreads();

  const bool active = v_col < config.value_width;
  const float decay = expf(log_decay[value_head]);
  const float beta_val = beta[value_head];

  float prediction = 0.0F;
  if (active) {
#pragma unroll 1
    for (std::uint32_t key_lane = static_cast<std::uint32_t>(k_tid);
         key_lane < config.key_width;
         key_lane += static_cast<std::uint32_t>(key_threads)) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          v_col;
      const float decayed = __fmul_rn(source[index], decay);
      prediction = __fadd_rn(
          prediction,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), decayed));
    }
  }
  reduce_buf[threadIdx.x] = prediction;
  __syncthreads();
  for (int stride = key_threads / 2; stride > 0; stride >>= 1) {
    if (k_tid < stride) {
      reduce_buf[threadIdx.x] = __fadd_rn(
          reduce_buf[threadIdx.x],
          reduce_buf[threadIdx.x +
                     static_cast<unsigned int>(stride) * value_tile]);
    }
    __syncthreads();
  }
  const float kv_acc = reduce_buf[threadIdx.x % value_tile];
  const float delta =
      active ? __fmul_rn(value[v_col] - kv_acc, beta_val) : 0.0F;

  float attn_shard = 0.0F;
  if (active) {
#pragma unroll 1
    for (std::uint32_t key_lane = static_cast<std::uint32_t>(k_tid);
         key_lane < config.key_width;
         key_lane += static_cast<std::uint32_t>(key_threads)) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          v_col;
      const float decayed = __fmul_rn(source[index], decay);
      const float updated = __fadd_rn(
          decayed, __fmul_rn(__fmul_rn(key[key_lane], key_inverse), delta));
      candidate[index] = updated;
      attn_shard = __fadd_rn(
          attn_shard,
          __fmul_rn(__fmul_rn(query[key_lane], query_inverse), updated));
    }
  }
  reduce_buf[threadIdx.x] = attn_shard;
  __syncthreads();
  for (int stride = key_threads / 2; stride > 0; stride >>= 1) {
    if (k_tid < stride) {
      reduce_buf[threadIdx.x] = __fadd_rn(
          reduce_buf[threadIdx.x],
          reduce_buf[threadIdx.x +
                     static_cast<unsigned int>(stride) * value_tile]);
    }
    __syncthreads();
  }
  if (k_tid == 0 && active) {
    output[static_cast<std::size_t>(value_head) * config.value_width + v_col] =
        reduce_buf[threadIdx.x];
  }
}

inline unsigned int gdn_decode_grid_y(const GdnConfig& config,
                                      unsigned int tile) noexcept {
  if (tile == 0 || config.value_width == 0) return 0;
  return (config.value_width + tile - 1U) / tile;
}

cudaError_t launch_gdn_decode_tiled_recurrence(
    const GdnConfig& config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, bool value_is_tiled,
    cudaStream_t stream) noexcept {
  const unsigned int tile = gdn_decode_value_tile();
  if (tile != kGdnDecodeValueTile16 && tile != kGdnDecodeValueTile32) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(config.value_heads, gdn_decode_grid_y(config, tile));
  const dim3 block(kGdnDecodeThreads);
  if (tile == kGdnDecodeValueTile16) {
    prepare_recurrence_decode_tiled<kGdnDecodeValueTile16>
        <<<grid, block, 0, stream>>>(config, convolution_output, log_decay, beta,
                                     source, candidate, output, value_is_tiled);
    record_gdn_decode_launch(kGdnDecodeLaunchVariantTile16, tile, grid.x,
                             grid.y);
  } else {
    prepare_recurrence_decode_tiled<kGdnDecodeValueTile32>
        <<<grid, block, 0, stream>>>(config, convolution_output, log_decay, beta,
                                     source, candidate, output, value_is_tiled);
    record_gdn_decode_launch(kGdnDecodeLaunchVariantTile32, tile, grid.x,
                             grid.y);
  }
  return cudaPeekAtLastError();
}

// Module-static 32-float inverse scratch. Not session-owned; 16 query + 16 key
// RMS inverses for GdnConfig{16,48,128,128,4}. Logical vs device GDN state
// counts stay identical (48*128*128).
__device__ float g_gdn_decode_qk_inverses[kGdnDecodeQkInverseCount];

__global__ void prepare_decode_qk_inverses(GdnConfig config,
                                           const float* convolution_output) {
  const std::uint32_t key_head = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  if (key_head >= config.key_heads || lane >= 32) return;
  const float* query = convolution_output + key_head * config.key_width;
  const float* key = convolution_output +
                     static_cast<std::size_t>(config.key_heads) * config.key_width +
                     key_head * config.key_width;
  float query_squares = 0.0F;
  float key_squares = 0.0F;
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const int index = row * 32 + lane;
    const float q = query[index];
    const float k = key[index];
    query_squares = __fadd_rn(query_squares, __fmul_rn(q, q));
    key_squares = __fadd_rn(key_squares, __fmul_rn(k, k));
  }
  query_squares = gdn_warp_sum(query_squares);
  key_squares = gdn_warp_sum(key_squares);
  if (lane == 0) {
    g_gdn_decode_qk_inverses[key_head] =
        1.0F / sqrtf(query_squares + kDecodeGdnL2Epsilon) /
        sqrtf(static_cast<float>(config.key_width));
    g_gdn_decode_qk_inverses[config.key_heads + key_head] =
        1.0F / sqrtf(key_squares + kDecodeGdnL2Epsilon);
  }
}

__global__ void gdn_relayout_col_to_row_inplace(float* state,
                                                std::uint32_t heads,
                                                std::uint32_t width) {
  extern __shared__ float tile[];
  const std::uint32_t head = blockIdx.x;
  if (head >= heads) return;
  float* base = state + static_cast<std::size_t>(head) * width * width;
  for (std::uint32_t row = threadIdx.y; row < width; row += blockDim.y) {
    for (std::uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
      tile[row * width + col] = base[static_cast<std::size_t>(col) * width + row];
    }
  }
  __syncthreads();
  for (std::uint32_t row = threadIdx.y; row < width; row += blockDim.y) {
    for (std::uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
      base[static_cast<std::size_t>(row) * width + col] = tile[row * width + col];
    }
  }
}

__global__ void gdn_relayout_row_to_col_inplace(float* state,
                                                std::uint32_t heads,
                                                std::uint32_t width) {
  extern __shared__ float tile[];
  const std::uint32_t head = blockIdx.x;
  if (head >= heads) return;
  float* base = state + static_cast<std::size_t>(head) * width * width;
  for (std::uint32_t row = threadIdx.y; row < width; row += blockDim.y) {
    for (std::uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
      tile[row * width + col] = base[static_cast<std::size_t>(row) * width + col];
    }
  }
  __syncthreads();
  for (std::uint32_t row = threadIdx.y; row < width; row += blockDim.y) {
    for (std::uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
      base[static_cast<std::size_t>(col) * width + row] = tile[row * width + col];
    }
  }
}

inline cudaError_t gdn_set_relayout_dynamic_smem(std::size_t smem) noexcept {
  cudaError_t attr = cudaFuncSetAttribute(
      gdn_relayout_col_to_row_inplace,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (attr != cudaSuccess) return attr;
  return cudaFuncSetAttribute(
      gdn_relayout_row_to_col_inplace,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
}

// Production specialization: width 128, 48 value heads, four warps/CTA,
// grid=(48,32). Each warp owns one output column; each lane owns four key rows.
// llama.cpp gated_delta_net_cuda<128> at cc83d7b motivates the layout and
// register shard; arithmetic stays Quartz FP32 recurrence.
__global__ void __launch_bounds__(kGdnDecodeThreads, 2)
prepare_recurrence_decode_transposed(GdnConfig config,
                                     const float* convolution_output,
                                     const float* log_decay, const float* beta,
                                     const float* source, float* candidate,
                                     float* output, bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const std::uint32_t col = blockIdx.y * kGdnDecodeWarpsPerCta + threadIdx.y;
  if (value_head >= config.value_heads || col >= config.value_width) return;

  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::uint32_t key_head = value_head / reuse;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  const std::size_t head_base = static_cast<std::size_t>(value_head) *
                                config.key_width * config.value_width;
  const std::uint32_t replica = value_head % reuse;
  const std::uint32_t tiled_head = replica * config.key_heads + key_head;
  const std::uint32_t source_head = value_is_tiled ? tiled_head : value_head;
  const float* query = convolution_output + key_head * config.key_width;
  const float* key =
      convolution_output + query_count + key_head * config.key_width;
  const float* value = convolution_output + 2 * query_count +
                       source_head * config.value_width;
  const float query_inverse = g_gdn_decode_qk_inverses[key_head];
  const float key_inverse =
      g_gdn_decode_qk_inverses[config.key_heads + key_head];

  float s_shard[4];
  float q_reg[4];
  float k_reg[4];
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const std::uint32_t key_lane =
        static_cast<std::uint32_t>(row * 32 + lane);
    s_shard[row] = source[head_base + static_cast<std::size_t>(col) *
                                          config.key_width +
                          key_lane];
    q_reg[row] = __fmul_rn(query[key_lane], query_inverse);
    k_reg[row] = __fmul_rn(key[key_lane], key_inverse);
  }

  const float decay = expf(log_decay[value_head]);
  const float beta_val = beta[value_head];
  float kv_shard = 0.0F;
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    kv_shard = __fadd_rn(kv_shard, __fmul_rn(s_shard[row], k_reg[row]));
  }
  const float prediction = __fmul_rn(gdn_warp_sum(kv_shard), decay);
  const float delta = __fmul_rn(value[col] - prediction, beta_val);

  float attn_shard = 0.0F;
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const float decayed = __fmul_rn(s_shard[row], decay);
    const float updated = __fadd_rn(decayed, __fmul_rn(k_reg[row], delta));
    s_shard[row] = updated;
    attn_shard = __fadd_rn(attn_shard, __fmul_rn(q_reg[row], updated));
    const std::uint32_t key_lane =
        static_cast<std::uint32_t>(row * 32 + lane);
    candidate[head_base + static_cast<std::size_t>(col) * config.key_width +
              key_lane] = updated;
  }
  const float attn = gdn_warp_sum(attn_shard);
  if (lane == 0) {
    output[static_cast<std::size_t>(value_head) * config.value_width + col] =
        attn;
  }
}

inline bool gdn_decode_transposed_shape(const GdnConfig& config) noexcept {
  return config.key_heads == 16 && config.value_heads == 48 &&
         config.key_width == 128 && config.value_width == 128;
}

cudaError_t launch_gdn_decode_transposed_recurrence(
    const GdnConfig& config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, bool value_is_tiled,
    cudaStream_t stream) noexcept {
  if (!gdn_decode_transposed_shape(config) || convolution_output == nullptr ||
      log_decay == nullptr || beta == nullptr || source == nullptr ||
      candidate == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(config.value_heads, kGdnDecodeTransposedGridY);
  const dim3 block(32, kGdnDecodeWarpsPerCta);
  const std::size_t smem =
      static_cast<std::size_t>(config.value_width) * config.value_width *
      sizeof(float);
  cudaError_t error = gdn_set_relayout_dynamic_smem(smem);
  if (error != cudaSuccess) return error;
  if (source != candidate) {
    error = launch_gdn_transpose_state_in(config, source, candidate, stream);
    if (error != cudaSuccess) return error;
  } else {
    gdn_relayout_row_to_col_inplace<<<config.value_heads, block, smem, stream>>>(
        candidate, config.value_heads, config.value_width);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  prepare_decode_qk_inverses<<<config.key_heads, 32, 0, stream>>>(
      config, convolution_output);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  prepare_recurrence_decode_transposed<<<grid, block, 0, stream>>>(
      config, convolution_output, log_decay, beta, candidate, candidate, output,
      value_is_tiled);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  gdn_relayout_col_to_row_inplace<<<config.value_heads, block, smem, stream>>>(
      candidate, config.value_heads, config.value_width);
  error = cudaPeekAtLastError();
  if (error == cudaSuccess) {
    record_gdn_decode_launch(kGdnDecodeLaunchVariantTransposed, 0, grid.x,
                             grid.y);
  }
  return error;
}
