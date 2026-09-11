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
