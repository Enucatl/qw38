#include "attention_decode.h"

#include <cmath>

namespace qw38::cuda {
namespace {

constexpr std::uint32_t kMaximumQueryHeads = 24;
constexpr std::uint32_t kMaximumKvHeads = 4;
constexpr std::uint32_t kMaximumHeadWidth = 256;
constexpr int kThreads = 256;
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kRopeTheta = 10000000.0F;

bool valid_config(const AttentionConfig& config) noexcept {
  return config.query_heads > 0 &&
         config.query_heads <= kMaximumQueryHeads && config.kv_heads > 0 &&
         config.kv_heads <= kMaximumKvHeads && config.head_width > 0 &&
         config.head_width <= kMaximumHeadWidth && config.capacity > 0 &&
         config.query_heads % config.kv_heads == 0 &&
         config.rotary_width <= config.head_width &&
         config.rotary_width % 2 == 0;
}

__global__ void normalize_query(AttentionConfig config, std::size_t position,
                                const float* query, const float* scale,
                                float* normalized) {
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  __shared__ float inverse;
  if (lane == 0) {
    float sum = 0.0F;
    for (std::uint32_t index = 0; index < config.head_width; ++index) {
      const float item = query[head * config.head_width + index];
      sum = __fadd_rn(sum, __fmul_rn(item, item));
    }
    inverse = 1.0F /
              sqrtf(sum / static_cast<float>(config.head_width) + kRmsEpsilon);
  }
  __syncthreads();
  if (lane < config.head_width) {
    normalized[head * config.head_width + lane] =
        __fmul_rn(__fmul_rn(query[head * config.head_width + lane], inverse),
                   scale[lane]);
  }
  __syncthreads();
  const std::uint32_t half = config.rotary_width / 2;
  if (lane < half) {
    const std::size_t base =
        static_cast<std::size_t>(head) * config.head_width;
    const float first = normalized[base + lane];
    const float second = normalized[base + half + lane];
    const float exponent =
        static_cast<float>(lane * 2) / static_cast<float>(config.rotary_width);
    const float angle =
        static_cast<float>(position) / powf(kRopeTheta, exponent);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    normalized[base + lane] =
        __fsub_rn(__fmul_rn(first, cosine), __fmul_rn(second, sine));
    normalized[base + half + lane] =
        __fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine));
  }
}

__global__ void normalize_key_and_stage_value(
    AttentionConfig config, std::size_t position, const float* key,
    const float* value, const float* scale, float* normalized,
    __nv_bfloat16* candidate_key, __nv_bfloat16* candidate_value) {
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  __shared__ float inverse;
  if (lane == 0) {
    float sum = 0.0F;
    for (std::uint32_t index = 0; index < config.head_width; ++index) {
      const float item = key[head * config.head_width + index];
      sum = __fadd_rn(sum, __fmul_rn(item, item));
    }
    inverse = 1.0F /
              sqrtf(sum / static_cast<float>(config.head_width) + kRmsEpsilon);
  }
  __syncthreads();
  const std::size_t base =
      static_cast<std::size_t>(head) * config.head_width;
  if (lane < config.head_width) {
    normalized[base + lane] =
        __fmul_rn(__fmul_rn(key[base + lane], inverse), scale[lane]);
    candidate_value[base + lane] = __float2bfloat16_rn(value[base + lane]);
  }
  __syncthreads();
  const std::uint32_t half = config.rotary_width / 2;
  if (lane < half) {
    const float first = normalized[base + lane];
    const float second = normalized[base + half + lane];
    const float exponent =
        static_cast<float>(lane * 2) / static_cast<float>(config.rotary_width);
    const float angle =
        static_cast<float>(position) / powf(kRopeTheta, exponent);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    normalized[base + lane] =
        __fsub_rn(__fmul_rn(first, cosine), __fmul_rn(second, sine));
    normalized[base + half + lane] =
        __fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine));
  }
  __syncthreads();
  if (lane < config.head_width) {
    candidate_key[base + lane] = __float2bfloat16_rn(normalized[base + lane]);
  }
}

__global__ void grouped_attention(
    AttentionConfig config, std::size_t chunk_start, std::size_t token_offset,
    std::size_t score_stride, const float* query,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* scores, float* output) {
  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const std::size_t position = chunk_start + token_offset;
  const std::uint32_t group_size = config.query_heads / config.kv_heads;
  const std::uint32_t kv_head = query_head / group_size;
  const std::size_t query_base =
      static_cast<std::size_t>(query_head) * config.head_width;
  const std::size_t row_values =
      static_cast<std::size_t>(config.kv_heads) * config.head_width;
  float* head_scores = scores + query_head * score_stride;
  __shared__ float denominator;
  if (lane == 0) {
    float maximum = -INFINITY;
    const float scaling =
        1.0F / sqrtf(static_cast<float>(config.head_width));
    for (std::size_t context = 0; context <= position; ++context) {
      const __nv_bfloat16* key_row =
          context < chunk_start
              ? committed_key + context * row_values
              : candidate_key + (context - chunk_start) * row_values;
      const std::size_t key_base =
          static_cast<std::size_t>(kv_head) * config.head_width;
      float score = 0.0F;
      for (std::uint32_t index = 0; index < config.head_width; ++index) {
        score = __fadd_rn(
            score,
            __fmul_rn(query[query_base + index],
                      __bfloat162float(key_row[key_base + index])));
      }
      head_scores[context] = __fmul_rn(score, scaling);
      maximum = fmaxf(maximum, head_scores[context]);
    }
    denominator = 0.0F;
    for (std::size_t context = 0; context <= position; ++context) {
      head_scores[context] = expf(head_scores[context] - maximum);
      denominator = __fadd_rn(denominator, head_scores[context]);
    }
  }
  __syncthreads();
  if (lane < config.head_width) {
    float result = 0.0F;
    const std::size_t value_base =
        static_cast<std::size_t>(kv_head) * config.head_width;
    for (std::size_t context = 0; context <= position; ++context) {
      const __nv_bfloat16* value_row =
          context < chunk_start
              ? committed_value + context * row_values
              : candidate_value + (context - chunk_start) * row_values;
      result = __fadd_rn(
          result,
          __fmul_rn(head_scores[context] / denominator,
                    __bfloat162float(value_row[value_base + lane])));
    }
    const float gate_value = gate[query_base + lane];
    const float sigmoid =
        gate_value >= 0.0F
            ? 1.0F / (1.0F + expf(-gate_value))
            : expf(gate_value) / (1.0F + expf(gate_value));
    output[query_base + lane] = __fmul_rn(result, sigmoid);
  }
}

__global__ void commit_rows(const __nv_bfloat16* candidate_key,
                            const __nv_bfloat16* candidate_value,
                            __nv_bfloat16* committed_key,
                            __nv_bfloat16* committed_value,
                            std::size_t row_values,
                            std::size_t start_position,
                            std::size_t token_count,
                            std::uint64_t new_frontier,
                            std::uint64_t* committed_frontier) {
  const std::size_t values = token_count * row_values;
  const std::size_t target = start_position * row_values;
  for (std::size_t index = threadIdx.x; index < values;
       index += blockDim.x) {
    committed_key[target + index] = candidate_key[index];
    committed_value[target + index] = candidate_value[index];
  }
  __syncthreads();
  if (threadIdx.x == 0) *committed_frontier = new_frontier;
}

}  // namespace

std::size_t attention_query_values(const AttentionConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return static_cast<std::size_t>(config.query_heads) * config.head_width;
}

std::size_t attention_kv_row_values(const AttentionConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return static_cast<std::size_t>(config.kv_heads) * config.head_width;
}

std::size_t attention_cache_values(const AttentionConfig& config) noexcept {
  return attention_kv_row_values(config) * config.capacity;
}

std::size_t attention_score_values(const AttentionConfig& config,
                                   std::size_t position) noexcept {
  if (!valid_config(config) || position >= config.capacity) return 0;
  return static_cast<std::size_t>(config.query_heads) * (position + 1);
}

std::size_t attention_chunk_score_values(const AttentionConfig& config,
                                         std::size_t start_position,
                                         std::size_t token_count) noexcept {
  if (!valid_config(config) || token_count == 0 ||
      start_position >= config.capacity ||
      token_count > config.capacity - start_position) {
    return 0;
  }
  return static_cast<std::size_t>(config.query_heads) *
         (start_position + token_count);
}

cudaError_t launch_attention_prepare(
    const AttentionConfig& config, std::size_t position, const float* query,
    const float* key, const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_row,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept {
  return launch_attention_prepare_chunk(
      config, position, 1, query, key, value, query_norm_scale, key_norm_scale,
      output_gate, committed, candidate_row, normalized_query, normalized_key,
      score_workspace, output, stream);
}

cudaError_t launch_attention_prepare_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept {
  const std::size_t score_values = attention_chunk_score_values(
      config, start_position, token_count);
  if (score_values == 0 || query == nullptr ||
      key == nullptr || value == nullptr || query_norm_scale == nullptr ||
      key_norm_scale == nullptr || output_gate == nullptr ||
      committed.key == nullptr || committed.value == nullptr ||
      candidate_rows.key == nullptr || candidate_rows.value == nullptr ||
      normalized_query == nullptr || normalized_key == nullptr ||
      score_workspace == nullptr || output == nullptr ||
      candidate_rows.key == committed.key ||
      candidate_rows.value == committed.value) {
    return cudaErrorInvalidValue;
  }
  const std::size_t query_values = attention_query_values(config);
  const std::size_t row_values = attention_kv_row_values(config);
  const std::size_t score_stride = start_position + token_count;
  for (std::size_t token = 0; token < token_count; ++token) {
    const std::size_t position = start_position + token;
    normalize_query<<<config.query_heads, kThreads, 0, stream>>>(
        config, position, query + token * query_values, query_norm_scale,
        normalized_query);
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    normalize_key_and_stage_value<<<config.kv_heads, kThreads, 0, stream>>>(
        config, position, key + token * row_values,
        value + token * row_values, key_norm_scale, normalized_key,
        candidate_rows.key + token * row_values,
        candidate_rows.value + token * row_values);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    grouped_attention<<<config.query_heads, kThreads, 0, stream>>>(
        config, start_position, token, score_stride, normalized_query,
        output_gate + token * query_values, committed.key, committed.value,
        candidate_rows.key, candidate_rows.value, score_workspace,
        output + token * query_values);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

cudaError_t launch_attention_commit(
    const AttentionConfig& config, std::size_t position,
    const AttentionCache& candidate_row, const AttentionCache& committed,
    std::uint64_t new_frontier, std::uint64_t* committed_frontier,
    cudaStream_t stream) noexcept {
  return launch_attention_commit_chunk(
      config, position, 1, candidate_row, committed, new_frontier,
      committed_frontier, stream);
}

cudaError_t launch_attention_commit_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const AttentionCache& candidate_rows,
    const AttentionCache& committed, std::uint64_t new_frontier,
    std::uint64_t* committed_frontier, cudaStream_t stream) noexcept {
  const std::size_t row_values = attention_kv_row_values(config);
  if (row_values == 0 || token_count == 0 ||
      start_position >= config.capacity ||
      token_count > config.capacity - start_position ||
      candidate_rows.key == nullptr || candidate_rows.value == nullptr ||
      committed.key == nullptr || committed.value == nullptr ||
      committed_frontier == nullptr || candidate_rows.key == committed.key ||
      candidate_rows.value == committed.value) {
    return cudaErrorInvalidValue;
  }
  commit_rows<<<1, kThreads, 0, stream>>>(
      candidate_rows.key, candidate_rows.value, committed.key, committed.value,
      row_values, start_position, token_count, new_frontier,
      committed_frontier);
  return cudaPeekAtLastError();
}

}  // namespace qw38::cuda
