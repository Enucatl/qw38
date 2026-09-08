#include "attention_decode.h"
#include "mma.cuh"

#include <cmath>
#include <cstdint>

#include <cuda_fp16.h>

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
              ? committed_key + attention_kv_physical_index(
                                    context, kv_head, 0, config.capacity,
                                    config.head_width)
              : candidate_key + (context - chunk_start) * row_values +
                    static_cast<std::size_t>(kv_head) * config.head_width;
      float score = 0.0F;
      for (std::uint32_t index = 0; index < config.head_width; ++index) {
        score = __fadd_rn(
            score,
            __fmul_rn(query[query_base + index],
                      __bfloat162float(key_row[index])));
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
    for (std::size_t context = 0; context <= position; ++context) {
      const __nv_bfloat16* value_row =
          context < chunk_start
              ? committed_value + attention_kv_physical_index(
                                      context, kv_head, 0, config.capacity,
                                      config.head_width)
              : candidate_value + (context - chunk_start) * row_values +
                    static_cast<std::size_t>(kv_head) * config.head_width;
      result = __fadd_rn(
          result,
          __fmul_rn(head_scores[context] / denominator,
                    __bfloat162float(value_row[lane])));
    }
    const float gate_value = gate[query_base + lane];
    const float sigmoid =
        gate_value >= 0.0F
            ? 1.0F / (1.0F + expf(-gate_value))
            : expf(gate_value) / (1.0F + expf(gate_value));
    output[query_base + lane] = __fmul_rn(result, sigmoid);
  }
}

// The chunk path deliberately keeps the KV tile private to one query row/head.
// This is the OPT-006/007 seam: correctness and launch shape are established
// here without changing either GQA reuse or block ownership.
__global__ void stage_chunk_rows(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* key, const float* value, const float* scale,
    __nv_bfloat16* candidate_key, __nv_bfloat16* candidate_value,
    float* normalized_key) {
  const std::uint32_t kv_head = blockIdx.x;
  const std::size_t token = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (token >= token_count) return;
  const std::size_t width = config.head_width;
  const std::size_t base = token * config.kv_heads * width + kv_head * width;
  __shared__ float normalized[kMaximumHeadWidth];
  __shared__ float inverse;
  if (lane == 0) {
    float sum = 0.0F;
    for (std::uint32_t i = 0; i < config.head_width; ++i) {
      const float item = key[base + i];
      sum = __fadd_rn(sum, __fmul_rn(item, item));
    }
    inverse = 1.0F / sqrtf(sum / static_cast<float>(width) + kRmsEpsilon);
  }
  __syncthreads();
  if (lane < width) normalized[lane] = __fmul_rn(key[base + lane] * inverse, scale[lane]);
  __syncthreads();
  const std::uint32_t half = config.rotary_width / 2;
  if (lane < half) {
    const float first = normalized[lane];
    const float second = normalized[half + lane];
    const float exponent = static_cast<float>(lane * 2) / config.rotary_width;
    const float angle = static_cast<float>(start_position + token) / powf(kRopeTheta, exponent);
    const float c = cosf(angle), s = sinf(angle);
    normalized[lane] = __fsub_rn(first * c, second * s);
    normalized[half + lane] = __fadd_rn(second * c, first * s);
  }
  __syncthreads();
  if (lane < width) {
    if (token_count == 1) normalized_key[base + lane] = normalized[lane];
    candidate_key[base + lane] = __float2bfloat16_rn(normalized[lane]);
    candidate_value[base + lane] = __float2bfloat16_rn(value[base + lane]);
  }
}

template <bool RecordSpans>
__device__ void load_kv_tile_rows(
    AttentionConfig config, std::size_t start_position, std::size_t tile,
    std::size_t rows, std::uint32_t kv_head, std::uint32_t lane,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    __nv_bfloat16* keys, __nv_bfloat16* values, AttentionKvTileSpan* records,
    std::uint32_t* record_count, std::uint32_t record_capacity) {
  const std::size_t width = config.head_width;
  const std::size_t row_values =
      static_cast<std::size_t>(config.kv_heads) * width;
  if (tile + rows <= start_position) {
    const std::size_t base = attention_kv_tile_base_offset(
        kv_head, tile, config.capacity, config.head_width);
    const __nv_bfloat16* tile_base = committed_key + base;
    const __nv_bfloat16* tile_base_v = committed_value + base;
    if (lane < width) {
      for (std::size_t row = 0; row < rows; ++row) {
        keys[row * kMaximumHeadWidth + lane] = tile_base[row * width + lane];
        values[row * kMaximumHeadWidth + lane] =
            tile_base_v[row * width + lane];
      }
    }
    if constexpr (RecordSpans) {
      if (lane == 0 && records != nullptr && record_count != nullptr) {
        const unsigned slot = atomicAdd(
            reinterpret_cast<unsigned int*>(record_count), 1U);
        if (slot < record_capacity) {
          records[slot].first_address =
              static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tile_base));
          records[slot].last_address = static_cast<std::uint64_t>(
              reinterpret_cast<std::uintptr_t>(tile_base + rows * width - 1));
          records[slot].rows = static_cast<std::uint32_t>(rows);
          records[slot].kv_head = kv_head;
          records[slot].tile_start = static_cast<std::uint32_t>(tile);
          records[slot].coalesced = 1;
        }
      }
    }
  } else {
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t absolute = tile + row;
      const __nv_bfloat16* ksrc;
      const __nv_bfloat16* vsrc;
      if (absolute < start_position) {
        const std::size_t offset = attention_kv_physical_index(
            absolute, kv_head, 0, config.capacity, config.head_width);
        ksrc = committed_key + offset;
        vsrc = committed_value + offset;
      } else {
        ksrc = candidate_key + (absolute - start_position) * row_values +
               kv_head * width;
        vsrc = candidate_value + (absolute - start_position) * row_values +
               kv_head * width;
      }
      if (lane < width) {
        keys[row * kMaximumHeadWidth + lane] = ksrc[lane];
        values[row * kMaximumHeadWidth + lane] = vsrc[lane];
      }
    }
    if constexpr (RecordSpans) {
      if (lane == 0 && records != nullptr && record_count != nullptr) {
        const unsigned slot = atomicAdd(
            reinterpret_cast<unsigned int*>(record_count), 1U);
        if (slot < record_capacity) {
          records[slot].first_address = 0;
          records[slot].last_address = 0;
          records[slot].rows = static_cast<std::uint32_t>(rows);
          records[slot].kv_head = kv_head;
          records[slot].tile_start = static_cast<std::uint32_t>(tile);
          records[slot].coalesced = 0;
        }
      }
    }
  }
}

template <bool CountLoads>
__global__ void per_query_tiled_chunk_attention(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query, std::uint64_t* kv_load_values) {
  const std::uint32_t query_head = blockIdx.x;
  const std::size_t token = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (token >= token_count) return;
  const std::size_t width = config.head_width;
  const std::size_t qbase = token * config.query_heads * width + query_head * width;
  const std::uint32_t group = config.query_heads / config.kv_heads;
  const std::uint32_t kv_head = query_head / group;
  extern __shared__ unsigned char raw[];
  float* q = reinterpret_cast<float*>(raw);
  __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(q + kMaximumHeadWidth);
  __nv_bfloat16* values = keys + kAttentionKvTileRows * kMaximumHeadWidth;
  __shared__ float inverse;
  __shared__ float score;
  if (lane == 0) {
    float sum = 0.0F;
    for (std::uint32_t i = 0; i < config.head_width; ++i) {
      const float item = query[qbase + i];
      sum = __fadd_rn(sum, item * item);
    }
    inverse = 1.0F / sqrtf(sum / static_cast<float>(width) + kRmsEpsilon);
  }
  __syncthreads();
  if (lane < width) q[lane] = query[qbase + lane] * inverse * query_scale[lane];
  __syncthreads();
  const std::uint32_t half = config.rotary_width / 2;
  if (lane < half) {
    const float first = q[lane], second = q[half + lane];
    const float exponent = static_cast<float>(lane * 2) / config.rotary_width;
    const float angle = static_cast<float>(start_position + token) / powf(kRopeTheta, exponent);
    const float c = cosf(angle), s = sinf(angle);
    q[lane] = first * c - second * s;
    q[half + lane] = second * c + first * s;
  }
  __syncthreads();
  if (token_count == 1 && lane < width)
    normalized_query[query_head * width + lane] = q[lane];
  float maximum = -INFINITY, denominator = 0.0F;
  float accumulator = 0.0F;
  std::uint64_t local_loads = 0;
  const std::size_t position = start_position + token;
  for (std::size_t tile = 0; tile <= position; tile += kAttentionKvTileRows) {
    const std::size_t rows =
        (position + 1 - tile) < kAttentionKvTileRows
            ? (position + 1 - tile)
            : kAttentionKvTileRows;
    load_kv_tile_rows<false>(
        config, start_position, tile, rows, kv_head, lane, committed_key,
        committed_value, candidate_key, candidate_value, keys, values, nullptr,
        nullptr, 0);
    if constexpr (CountLoads) {
      if (lane < width) local_loads += 2 * rows;
    }
    __syncthreads();
    for (std::size_t row = 0; row < rows; ++row) {
      if (lane == 0) {
        float dot = 0.0F;
        for (std::uint32_t i = 0; i < config.head_width; ++i)
          dot = __fadd_rn(dot, q[i] * __bfloat162float(keys[row * kMaximumHeadWidth + i]));
        score = dot / sqrtf(static_cast<float>(width));
      }
      __syncthreads();
      const float old_max = maximum;
      maximum = fmaxf(maximum, score);
      const float rescale = old_max == -INFINITY ? 0.0F : expf(old_max - maximum);
      denominator = denominator * rescale + expf(score - maximum);
      accumulator = accumulator * rescale + expf(score - maximum) * __bfloat162float(values[row * kMaximumHeadWidth + lane]);
      __syncthreads();
    }
    __syncthreads();
  }
  if (lane < width) {
    const float g = gate[qbase + lane];
    const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g)) : expf(g) / (1.0F + expf(g));
    output[qbase + lane] = (accumulator / denominator) * sigmoid;
  }
  if constexpr (CountLoads) {
    if (local_loads != 0)
      atomicAdd(reinterpret_cast<unsigned long long*>(kv_load_values),
                static_cast<unsigned long long>(local_loads));
  }
}

template <bool CountLoads>
__global__ void one_row_grouped_tiled_chunk_attention(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query, std::uint64_t* kv_load_values) {
  constexpr std::uint32_t kMaximumGroup =
      kMaximumQueryHeads / kMaximumKvHeads;
  const std::uint32_t kv_head = blockIdx.x;
  const std::size_t token = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (token >= token_count) return;
  const std::size_t width = config.head_width;
  const std::uint32_t group = config.query_heads / config.kv_heads;
  const std::uint32_t first_head = kv_head * group;
  extern __shared__ unsigned char raw[];
  float* scratch = reinterpret_cast<float*>(raw);
  __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(scratch + kMaximumHeadWidth);
  __nv_bfloat16* values = keys + kAttentionKvTileRows * kMaximumHeadWidth;
  __shared__ float inverse;
  __shared__ float score;
  float q[kMaximumGroup]{};
  float maximum[kMaximumGroup];
  float denominator[kMaximumGroup]{};
  float accumulator[kMaximumGroup]{};
  const std::uint32_t half = config.rotary_width / 2;
  for (std::uint32_t member = 0; member < group; ++member) {
    const std::uint32_t query_head = first_head + member;
    const std::size_t qbase =
        token * config.query_heads * width + query_head * width;
    if (lane == 0) {
      float sum = 0.0F;
      for (std::uint32_t i = 0; i < config.head_width; ++i) {
        const float item = query[qbase + i];
        sum = __fadd_rn(sum, item * item);
      }
      inverse = 1.0F / sqrtf(sum / static_cast<float>(width) + kRmsEpsilon);
    }
    __syncthreads();
    if (lane < width)
      scratch[lane] = query[qbase + lane] * inverse * query_scale[lane];
    __syncthreads();
    if (lane < half) {
      const float first = scratch[lane], second = scratch[half + lane];
      const float exponent = static_cast<float>(lane * 2) / config.rotary_width;
      const float angle = static_cast<float>(start_position + token) /
                          powf(kRopeTheta, exponent);
      const float c = cosf(angle), s = sinf(angle);
      scratch[lane] = first * c - second * s;
      scratch[half + lane] = second * c + first * s;
    }
    __syncthreads();
    if (lane < width) {
      q[member] = scratch[lane];
      if (token_count == 1)
        normalized_query[query_head * width + lane] = q[member];
    }
    maximum[member] = -INFINITY;
    __syncthreads();
  }
  std::uint64_t local_loads = 0;
  const std::size_t position = start_position + token;
  for (std::size_t tile = 0; tile <= position; tile += kAttentionKvTileRows) {
    const std::size_t rows =
        (position + 1 - tile) < kAttentionKvTileRows
            ? (position + 1 - tile)
            : kAttentionKvTileRows;
    load_kv_tile_rows<false>(
        config, start_position, tile, rows, kv_head, lane, committed_key,
        committed_value, candidate_key, candidate_value, keys, values, nullptr,
        nullptr, 0);
    if constexpr (CountLoads) {
      if (lane < width) local_loads += 2 * rows;
    }
    __syncthreads();
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::uint32_t member = 0; member < group; ++member) {
        if (lane < width)
          scratch[lane] =
              q[member] *
              __bfloat162float(keys[row * kMaximumHeadWidth + lane]);
        __syncthreads();
        if (lane == 0) {
          float dot = 0.0F;
          for (std::uint32_t i = 0; i < config.head_width; ++i)
            dot = __fadd_rn(dot, scratch[i]);
          score = dot / sqrtf(static_cast<float>(width));
        }
        __syncthreads();
        const float old_max = maximum[member];
        maximum[member] = fmaxf(maximum[member], score);
        const float rescale = old_max == -INFINITY
                                  ? 0.0F
                                  : expf(old_max - maximum[member]);
        denominator[member] = denominator[member] * rescale +
                              expf(score - maximum[member]);
        accumulator[member] =
            accumulator[member] * rescale +
            expf(score - maximum[member]) *
                __bfloat162float(values[row * kMaximumHeadWidth + lane]);
        __syncthreads();
      }
    }
  }
  if (lane < width) {
    for (std::uint32_t member = 0; member < group; ++member) {
      const std::uint32_t query_head = first_head + member;
      const std::size_t qbase =
          token * config.query_heads * width + query_head * width;
      const float g = gate[qbase + lane];
      const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g))
                                      : expf(g) / (1.0F + expf(g));
      output[qbase + lane] =
          (accumulator[member] / denominator[member]) * sigmoid;
    }
  }
  if constexpr (CountLoads) {
    if (local_loads != 0)
      atomicAdd(reinterpret_cast<unsigned long long*>(kv_load_values),
                static_cast<unsigned long long>(local_loads));
  }
}

template <bool RecordSpans>
__global__ void two_row_grouped_tiled_chunk_attention(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query, AttentionKvTileSpan* tile_spans,
    std::uint32_t* tile_span_count, std::uint32_t tile_span_capacity) {
  if constexpr (!RecordSpans) {
    (void)tile_spans;
    (void)tile_span_count;
    (void)tile_span_capacity;
  }
  constexpr std::uint32_t kMaximumGroup =
      kMaximumQueryHeads / kMaximumKvHeads;
  constexpr std::size_t kQueryRowsPerBlock = 2;
  const std::uint32_t kv_head = blockIdx.x;
  const std::size_t first_token = kQueryRowsPerBlock * blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (first_token >= token_count) return;
  const std::size_t active_rows =
      token_count - first_token < kQueryRowsPerBlock
          ? token_count - first_token
          : kQueryRowsPerBlock;
  const std::size_t width = config.head_width;
  const std::uint32_t group = config.query_heads / config.kv_heads;
  const std::uint32_t first_head = kv_head * group;
  extern __shared__ unsigned char raw[];
  float* scratch = reinterpret_cast<float*>(raw);
  __nv_bfloat16* keys =
      reinterpret_cast<__nv_bfloat16*>(scratch + kMaximumHeadWidth);
  __nv_bfloat16* values = keys + kAttentionKvTileRows * kMaximumHeadWidth;
  __shared__ float inverse[kQueryRowsPerBlock];
  __shared__ float score;
  float q[kQueryRowsPerBlock][kMaximumGroup]{};
  float maximum[kQueryRowsPerBlock][kMaximumGroup];
  float denominator[kQueryRowsPerBlock][kMaximumGroup]{};
  float accumulator[kQueryRowsPerBlock][kMaximumGroup]{};
  const std::uint32_t half = config.rotary_width / 2;
  for (std::size_t row = 0; row < active_rows; ++row) {
    const std::size_t token = first_token + row;
    for (std::uint32_t member = 0; member < group; ++member) {
      const std::uint32_t query_head = first_head + member;
      const std::size_t qbase =
          token * config.query_heads * width + query_head * width;
      if (lane == 0) {
        float sum = 0.0F;
        for (std::uint32_t i = 0; i < config.head_width; ++i) {
          const float item = query[qbase + i];
          sum = __fadd_rn(sum, item * item);
        }
        inverse[row] =
            1.0F / sqrtf(sum / static_cast<float>(width) + kRmsEpsilon);
      }
      __syncthreads();
      if (lane < width)
        scratch[lane] = query[qbase + lane] * inverse[row] * query_scale[lane];
      __syncthreads();
      if (lane < half) {
        const float first = scratch[lane], second = scratch[half + lane];
        const float exponent = static_cast<float>(lane * 2) / config.rotary_width;
        const float angle = static_cast<float>(start_position + token) /
                            powf(kRopeTheta, exponent);
        const float c = cosf(angle), s = sinf(angle);
        scratch[lane] = first * c - second * s;
        scratch[half + lane] = second * c + first * s;
      }
      __syncthreads();
      if (lane < width) {
        q[row][member] = scratch[lane];
        if (token_count == 1)
          normalized_query[query_head * width + lane] = q[row][member];
      }
      maximum[row][member] = -INFINITY;
      __syncthreads();
    }
  }
  const std::size_t last_position =
      start_position + first_token + active_rows - 1;
  for (std::size_t tile = 0; tile <= last_position;
       tile += kAttentionKvTileRows) {
    const std::size_t rows =
        (last_position + 1 - tile) < kAttentionKvTileRows
            ? (last_position + 1 - tile)
            : kAttentionKvTileRows;
    load_kv_tile_rows<RecordSpans>(
        config, start_position, tile, rows, kv_head, lane, committed_key,
        committed_value, candidate_key, candidate_value, keys, values,
        tile_spans, tile_span_count, tile_span_capacity);
    __syncthreads();
    for (std::size_t tile_row = 0; tile_row < rows; ++tile_row) {
      const std::size_t absolute = tile + tile_row;
      for (std::size_t row = 0; row < active_rows; ++row) {
        if (absolute > start_position + first_token + row) continue;
        for (std::uint32_t member = 0; member < group; ++member) {
          if (lane < width)
            scratch[lane] =
                q[row][member] * __bfloat162float(
                                      keys[tile_row * kMaximumHeadWidth + lane]);
          __syncthreads();
          if (lane == 0) {
            float dot = 0.0F;
            for (std::uint32_t i = 0; i < config.head_width; ++i)
              dot = __fadd_rn(dot, scratch[i]);
            score = dot / sqrtf(static_cast<float>(width));
          }
          __syncthreads();
          const float old_max = maximum[row][member];
          maximum[row][member] = fmaxf(maximum[row][member], score);
          const float rescale = old_max == -INFINITY
                                    ? 0.0F
                                    : expf(old_max - maximum[row][member]);
          denominator[row][member] =
              denominator[row][member] * rescale +
              expf(score - maximum[row][member]);
          accumulator[row][member] =
              accumulator[row][member] * rescale +
              expf(score - maximum[row][member]) *
                  __bfloat162float(
                      values[tile_row * kMaximumHeadWidth + lane]);
          __syncthreads();
        }
      }
    }
  }
  if (lane < width) {
    for (std::size_t row = 0; row < active_rows; ++row) {
      const std::size_t token = first_token + row;
      for (std::uint32_t member = 0; member < group; ++member) {
        const std::uint32_t query_head = first_head + member;
        const std::size_t qbase =
            token * config.query_heads * width + query_head * width;
        const float g = gate[qbase + lane];
        const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g))
                                        : expf(g) / (1.0F + expf(g));
        output[qbase + lane] =
            (accumulator[row][member] / denominator[row][member]) * sigmoid;
      }
    }
  }
}

template <int QueryRows>
__global__ void mma_grouped_chunk_attention(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query) {
  constexpr std::uint32_t kMaximumGroup =
      kMaximumQueryHeads / kMaximumKvHeads;
  const std::uint32_t kv_head = blockIdx.x;
  const std::size_t first_token =
      static_cast<std::size_t>(QueryRows) * blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  const int warp = threadIdx.x / 32;
  if (first_token >= token_count) return;
  const std::size_t active_rows =
      token_count - first_token < static_cast<std::size_t>(QueryRows)
          ? token_count - first_token
          : static_cast<std::size_t>(QueryRows);
  const std::size_t width = config.head_width;
  const std::uint32_t group = config.query_heads / config.kv_heads;
  const std::uint32_t first_head = kv_head * group;
  extern __shared__ unsigned char raw[];
  float* scratch = reinterpret_cast<float*>(raw);
  __nv_bfloat16* keys =
      reinterpret_cast<__nv_bfloat16*>(scratch + kMaximumHeadWidth);
  __nv_bfloat16* values = keys + kAttentionKvTileRows * kMaximumHeadWidth;
  __half* q_f16 = reinterpret_cast<__half*>(
      values + kAttentionKvTileRows * kMaximumHeadWidth);
  __half* q_lo_f16 = q_f16 + QueryRows * kMaximumHeadWidth;
  __half* k_f16 = q_lo_f16 + QueryRows * kMaximumHeadWidth;
  __shared__ float scores[QueryRows][8];
  __shared__ float inverse[QueryRows];
  float q[QueryRows][kMaximumGroup]{};
  float maximum[QueryRows][kMaximumGroup];
  float denominator[QueryRows][kMaximumGroup]{};
  float accumulator[QueryRows][kMaximumGroup]{};
  const std::uint32_t half = config.rotary_width / 2;
  for (std::size_t row = 0; row < active_rows; ++row) {
    const std::size_t token = first_token + row;
    for (std::uint32_t member = 0; member < group; ++member) {
      const std::uint32_t query_head = first_head + member;
      const std::size_t qbase =
          token * config.query_heads * width + query_head * width;
      if (lane == 0) {
        float sum = 0.0F;
        for (std::uint32_t i = 0; i < config.head_width; ++i) {
          const float item = query[qbase + i];
          sum = __fadd_rn(sum, item * item);
        }
        inverse[row] =
            1.0F / sqrtf(sum / static_cast<float>(width) + kRmsEpsilon);
      }
      __syncthreads();
      if (lane < width)
        scratch[lane] = query[qbase + lane] * inverse[row] * query_scale[lane];
      __syncthreads();
      if (lane < half) {
        const float first = scratch[lane], second = scratch[half + lane];
        const float exponent = static_cast<float>(lane * 2) / config.rotary_width;
        const float angle = static_cast<float>(start_position + token) /
                            powf(kRopeTheta, exponent);
        const float c = cosf(angle), s = sinf(angle);
        scratch[lane] = first * c - second * s;
        scratch[half + lane] = second * c + first * s;
      }
      __syncthreads();
      if (lane < width) {
        q[row][member] = scratch[lane];
        if (token_count == 1)
          normalized_query[query_head * width + lane] = q[row][member];
      }
      maximum[row][member] = -INFINITY;
      __syncthreads();
    }
  }
  const std::size_t last_position =
      start_position + first_token + active_rows - 1;
  const float scale = 1.0F / sqrtf(static_cast<float>(width));
  for (std::uint32_t member = 0; member < group; ++member) {
    if (lane < width) {
      for (std::size_t row = 0; row < static_cast<std::size_t>(QueryRows);
           ++row) {
        const float value = row < active_rows ? q[row][member] : 0.0F;
        const __half hi = __float2half_rn(value);
        q_f16[row * kMaximumHeadWidth + lane] = hi;
        q_lo_f16[row * kMaximumHeadWidth + lane] =
            __float2half_rn(value - __half2float(hi));
      }
    }
    __syncthreads();
    for (std::size_t tile = 0; tile <= last_position;
         tile += kAttentionKvTileRows) {
      const std::size_t rows =
          (last_position + 1 - tile) < kAttentionKvTileRows
              ? (last_position + 1 - tile)
              : kAttentionKvTileRows;
      load_kv_tile_rows<false>(
          config, start_position, tile, rows, kv_head, lane, committed_key,
          committed_value, candidate_key, candidate_value, keys, values,
          nullptr, nullptr, 0);
      __syncthreads();
      for (std::size_t pack = 0; pack < rows; pack += 8) {
        const std::size_t pack_rows =
            rows - pack < 8 ? rows - pack : 8;
        if (lane < width) {
          for (std::size_t kr = 0; kr < 8; ++kr) {
            const float kv =
                kr < pack_rows
                    ? __bfloat162float(keys[(pack + kr) * kMaximumHeadWidth +
                                            lane])
                    : 0.0F;
            k_f16[kr * kMaximumHeadWidth + lane] = __float2half_rn(kv);
          }
        }
        __syncthreads();
        if (warp == 0) {
          float cfrag[4] = {0.0F, 0.0F, 0.0F, 0.0F};
          float cfrag_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
          const int mma_lane = lane;
          for (int k0 = 0; k0 < static_cast<int>(width); k0 += 16) {
            std::uint32_t a[4];
            std::uint32_t a_lo[4];
            std::uint32_t b[2];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
              const int i = mma::tile16x8_half2_i(mma_lane, l);
              const int j = mma::tile16x8_half2_j(mma_lane, l);
              const __half* hi =
                  q_f16 + i * kMaximumHeadWidth + k0 + j * 2;
              const __half* lo =
                  q_lo_f16 + i * kMaximumHeadWidth + k0 + j * 2;
              a[l] = *reinterpret_cast<const std::uint32_t*>(hi);
              a_lo[l] = *reinterpret_cast<const std::uint32_t*>(lo);
            }
#pragma unroll
            for (int l = 0; l < 2; ++l) {
              const int n = mma::tile8x8_i(mma_lane, l);
              const int j = mma::tile8x8_j(mma_lane, l);
              const __half* ptr =
                  k_f16 + n * kMaximumHeadWidth + k0 + j * 2;
              b[l] = *reinterpret_cast<const std::uint32_t*>(ptr);
            }
            mma::mma_m16n8k16_f16_f32(cfrag, a, b);
            mma::mma_m16n8k16_f16_f32(cfrag_lo, a_lo, b);
          }
#pragma unroll
          for (int l = 0; l < 4; ++l) {
            const int i = mma::tile16x8_i(mma_lane, l);
            const int j = mma::tile16x8_j(mma_lane, l);
            scores[i][j] = (cfrag[l] + cfrag_lo[l]) * scale;
          }
        }
        __syncthreads();
        for (std::size_t kr = 0; kr < pack_rows; ++kr) {
          const std::size_t absolute = tile + pack + kr;
          for (std::size_t row = 0; row < active_rows; ++row) {
            if (absolute > start_position + first_token + row) continue;
            const float score = scores[row][kr];
            const float old_max = maximum[row][member];
            maximum[row][member] = fmaxf(maximum[row][member], score);
            const float rescale = old_max == -INFINITY
                                      ? 0.0F
                                      : expf(old_max - maximum[row][member]);
            denominator[row][member] =
                denominator[row][member] * rescale +
                expf(score - maximum[row][member]);
            if (lane < width) {
              accumulator[row][member] =
                  accumulator[row][member] * rescale +
                  expf(score - maximum[row][member]) *
                      __bfloat162float(
                          values[(pack + kr) * kMaximumHeadWidth + lane]);
            }
          }
        }
        __syncthreads();
      }
    }
  }
  if (lane < width) {
    for (std::size_t row = 0; row < active_rows; ++row) {
      const std::size_t token = first_token + row;
      for (std::uint32_t member = 0; member < group; ++member) {
        const std::uint32_t query_head = first_head + member;
        const std::size_t qbase =
            token * config.query_heads * width + query_head * width;
        const float g = gate[qbase + lane];
        const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g))
                                        : expf(g) / (1.0F + expf(g));
        output[qbase + lane] =
            (accumulator[row][member] / denominator[row][member]) * sigmoid;
      }
    }
  }
}

unsigned scatter_block_count(std::size_t values) noexcept {
  const std::size_t needed =
      (values + static_cast<std::size_t>(kThreads) - 1) /
      static_cast<std::size_t>(kThreads);
  if (needed == 0) return 1U;
  constexpr std::size_t kMaxBlocks = 2048;
  return static_cast<unsigned>(needed < kMaxBlocks ? needed : kMaxBlocks);
}

__global__ void scatter_committed_rows(
    const __nv_bfloat16* candidate_key_base,
    const __nv_bfloat16* candidate_value_base, __nv_bfloat16* committed_key_base,
    __nv_bfloat16* committed_value_base, std::size_t candidate_layer_stride,
    std::size_t committed_layer_stride, std::size_t start_position,
    std::size_t token_count, std::uint32_t kv_heads, std::uint32_t capacity,
    std::uint32_t head_width, std::uint64_t new_frontier,
    std::uint64_t* committed_frontier) {
  const std::size_t layer = blockIdx.y;
  const __nv_bfloat16* candidate_key =
      candidate_key_base + layer * candidate_layer_stride;
  const __nv_bfloat16* candidate_value =
      candidate_value_base + layer * candidate_layer_stride;
  __nv_bfloat16* committed_key =
      committed_key_base + layer * committed_layer_stride;
  __nv_bfloat16* committed_value =
      committed_value_base + layer * committed_layer_stride;
  const std::size_t values =
      token_count * static_cast<std::size_t>(kv_heads) * head_width;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < values; index += stride) {
    const std::size_t row_values =
        static_cast<std::size_t>(kv_heads) * head_width;
    const std::size_t rel_token = index / row_values;
    const std::size_t remainder = index % row_values;
    const std::uint32_t kv_head =
        static_cast<std::uint32_t>(remainder / head_width);
    const std::uint32_t lane =
        static_cast<std::uint32_t>(remainder % head_width);
    const std::size_t destination = attention_kv_physical_index(
        start_position + rel_token, kv_head, lane, capacity, head_width);
    committed_key[destination] = candidate_key[index];
    committed_value[destination] = candidate_value[index];
  }
  if (committed_frontier != nullptr && blockIdx.x == 0 && blockIdx.y == 0 &&
      threadIdx.x == 0) {
    *committed_frontier = new_frontier;
  }
}

__global__ void pack_committed_kv_kernel(
    const __nv_bfloat16* physical, __nv_bfloat16* logical,
    std::size_t token_begin, std::size_t token_count, std::uint32_t kv_heads,
    std::uint32_t capacity, std::uint32_t head_width) {
  const std::size_t values =
      token_count * static_cast<std::size_t>(kv_heads) * head_width;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < values;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t row_values =
        static_cast<std::size_t>(kv_heads) * head_width;
    const std::size_t rel_token = index / row_values;
    const std::size_t remainder = index % row_values;
    const std::uint32_t kv_head =
        static_cast<std::uint32_t>(remainder / head_width);
    const std::uint32_t lane =
        static_cast<std::uint32_t>(remainder % head_width);
    logical[index] = physical[attention_kv_physical_index(
        token_begin + rel_token, kv_head, lane, capacity, head_width)];
  }
}

__global__ void unpack_committed_kv_kernel(
    const __nv_bfloat16* logical, __nv_bfloat16* physical,
    std::size_t token_begin, std::size_t token_count, std::uint32_t kv_heads,
    std::uint32_t capacity, std::uint32_t head_width) {
  const std::size_t values =
      token_count * static_cast<std::size_t>(kv_heads) * head_width;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < values;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t row_values =
        static_cast<std::size_t>(kv_heads) * head_width;
    const std::size_t rel_token = index / row_values;
    const std::size_t remainder = index % row_values;
    const std::uint32_t kv_head =
        static_cast<std::uint32_t>(remainder / head_width);
    const std::uint32_t lane =
        static_cast<std::uint32_t>(remainder % head_width);
    physical[attention_kv_physical_index(token_begin + rel_token, kv_head, lane,
                                         capacity, head_width)] =
        logical[index];
  }
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

cudaError_t launch_attention_prepare_chunk_reference(
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

namespace {

cudaError_t stage_and_validate_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept {
  const std::size_t score_values =
      attention_chunk_score_values(config, start_position, token_count);
  if (score_values == 0 || query == nullptr || key == nullptr ||
      value == nullptr || query_norm_scale == nullptr ||
      key_norm_scale == nullptr || output_gate == nullptr ||
      committed.key == nullptr || committed.value == nullptr ||
      candidate_rows.key == nullptr || candidate_rows.value == nullptr ||
      normalized_query == nullptr || normalized_key == nullptr ||
      score_workspace == nullptr || output == nullptr ||
      candidate_rows.key == committed.key ||
      candidate_rows.value == committed.value) {
    return cudaErrorInvalidValue;
  }
  dim3 staging(config.kv_heads, static_cast<unsigned>(token_count), 1);
  stage_chunk_rows<<<staging, kThreads, 0, stream>>>(
      config, start_position, token_count, key, value, key_norm_scale,
      candidate_rows.key, candidate_rows.value, normalized_key);
  return cudaPeekAtLastError();
}

}  // namespace

cudaError_t launch_attention_prepare_chunk_tiled(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept {
  cudaError_t error = stage_and_validate_chunk(
      config, start_position, token_count, query, key, value, query_norm_scale,
      key_norm_scale, output_gate, committed, candidate_rows, normalized_query,
      normalized_key, score_workspace, output, stream);
  if (error != cudaSuccess) return error;
  const std::size_t shared_tiled =
      (2 * kAttentionKvTileRows * kMaximumHeadWidth) * sizeof(__nv_bfloat16) +
      kMaximumHeadWidth * sizeof(float);
  dim3 attention(config.kv_heads,
                 static_cast<unsigned>((token_count + 1) / 2), 1);
  two_row_grouped_tiled_chunk_attention<false>
      <<<attention, kThreads, shared_tiled, stream>>>(
          config, start_position, token_count, query, query_norm_scale,
          output_gate, committed.key, committed.value, candidate_rows.key,
          candidate_rows.value, output, normalized_query, nullptr, nullptr, 0);
  return cudaPeekAtLastError();
}

cudaError_t launch_attention_prepare_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept {
  if (token_count < 16) {
    return launch_attention_prepare_chunk_tiled(
        config, start_position, token_count, query, key, value,
        query_norm_scale, key_norm_scale, output_gate, committed,
        candidate_rows, normalized_query, normalized_key, score_workspace,
        output, stream);
  }
  cudaError_t error = stage_and_validate_chunk(
      config, start_position, token_count, query, key, value, query_norm_scale,
      key_norm_scale, output_gate, committed, candidate_rows, normalized_query,
      normalized_key, score_workspace, output, stream);
  if (error != cudaSuccess) return error;
  const std::size_t shared_tiled =
      (2 * kAttentionKvTileRows * kMaximumHeadWidth) * sizeof(__nv_bfloat16) +
      kMaximumHeadWidth * sizeof(float);
  const std::size_t shared_mma =
      shared_tiled + (16 + 16 + 8) * kMaximumHeadWidth * sizeof(__half);
  dim3 attention(config.kv_heads,
                 static_cast<unsigned>((token_count + 15) / 16), 1);
  error = cudaFuncSetAttribute(
      mma_grouped_chunk_attention<16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_mma));
  if (error != cudaSuccess) return error;
  mma_grouped_chunk_attention<16>
      <<<attention, kThreads, shared_mma, stream>>>(
          config, start_position, token_count, query, query_norm_scale,
          output_gate, committed.key, committed.value, candidate_rows.key,
          candidate_rows.value, output, normalized_query);
  return cudaPeekAtLastError();
}

template <bool Grouped>
cudaError_t launch_instrumented_tiled(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, std::uint64_t* kv_load_values,
    cudaStream_t stream) noexcept {
  const std::size_t score_values =
      attention_chunk_score_values(config, start_position, token_count);
  if (score_values == 0 || query == nullptr || key == nullptr ||
      value == nullptr || query_norm_scale == nullptr ||
      key_norm_scale == nullptr || output_gate == nullptr ||
      committed.key == nullptr || committed.value == nullptr ||
      candidate_rows.key == nullptr || candidate_rows.value == nullptr ||
      normalized_query == nullptr || normalized_key == nullptr ||
      score_workspace == nullptr || output == nullptr ||
      kv_load_values == nullptr || candidate_rows.key == committed.key ||
      candidate_rows.value == committed.value)
    return cudaErrorInvalidValue;
  dim3 staging(config.kv_heads, static_cast<unsigned>(token_count), 1);
  stage_chunk_rows<<<staging, kThreads, 0, stream>>>(
      config, start_position, token_count, key, value, key_norm_scale,
      candidate_rows.key, candidate_rows.value, normalized_key);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  const std::size_t shared =
      (2 * kAttentionKvTileRows * kMaximumHeadWidth) * sizeof(__nv_bfloat16) +
      kMaximumHeadWidth * sizeof(float);
  dim3 attention(Grouped ? config.kv_heads : config.query_heads,
                 static_cast<unsigned>(token_count), 1);
  if constexpr (Grouped) {
    one_row_grouped_tiled_chunk_attention<true>
        <<<attention, kThreads, shared, stream>>>(
        config, start_position, token_count, query, query_norm_scale,
        output_gate, committed.key, committed.value, candidate_rows.key,
        candidate_rows.value, output, normalized_query, kv_load_values);
  } else {
    per_query_tiled_chunk_attention<true>
        <<<attention, kThreads, shared, stream>>>(
            config, start_position, token_count, query, query_norm_scale,
            output_gate, committed.key, committed.value, candidate_rows.key,
            candidate_rows.value, output, normalized_query, kv_load_values);
  }
  return cudaPeekAtLastError();
}

cudaError_t launch_attention_prepare_chunk_grouped_instrumented(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, std::uint64_t* kv_load_values,
    cudaStream_t stream) noexcept {
  return launch_instrumented_tiled<true>(
      config, start_position, token_count, query, key, value, query_norm_scale,
      key_norm_scale, output_gate, committed, candidate_rows, normalized_query,
      normalized_key, score_workspace, output, kv_load_values, stream);
}

cudaError_t launch_attention_prepare_chunk_per_query_tiled_reference(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, std::uint64_t* kv_load_values,
    cudaStream_t stream) noexcept {
  return launch_instrumented_tiled<false>(
      config, start_position, token_count, query, key, value, query_norm_scale,
      key_norm_scale, output_gate, committed, candidate_rows, normalized_query,
      normalized_key, score_workspace, output, kv_load_values, stream);
}

cudaError_t launch_attention_prepare_chunk_tile_span_instrumented(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, AttentionKvTileSpan* tile_spans,
    std::uint32_t* tile_span_count, std::uint32_t tile_span_capacity,
    cudaStream_t stream) noexcept {
  const std::size_t score_values =
      attention_chunk_score_values(config, start_position, token_count);
  if (score_values == 0 || query == nullptr || key == nullptr ||
      value == nullptr || query_norm_scale == nullptr ||
      key_norm_scale == nullptr || output_gate == nullptr ||
      committed.key == nullptr || committed.value == nullptr ||
      candidate_rows.key == nullptr || candidate_rows.value == nullptr ||
      normalized_query == nullptr || normalized_key == nullptr ||
      score_workspace == nullptr || output == nullptr ||
      tile_spans == nullptr || tile_span_count == nullptr ||
      tile_span_capacity == 0 || candidate_rows.key == committed.key ||
      candidate_rows.value == committed.value) {
    return cudaErrorInvalidValue;
  }
  dim3 staging(config.kv_heads, static_cast<unsigned>(token_count), 1);
  stage_chunk_rows<<<staging, kThreads, 0, stream>>>(
      config, start_position, token_count, key, value, key_norm_scale,
      candidate_rows.key, candidate_rows.value, normalized_key);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  dim3 attention(config.kv_heads,
                 static_cast<unsigned>((token_count + 1) / 2), 1);
  const std::size_t shared =
      (2 * kAttentionKvTileRows * kMaximumHeadWidth) * sizeof(__nv_bfloat16) +
      kMaximumHeadWidth * sizeof(float);
  two_row_grouped_tiled_chunk_attention<true>
      <<<attention, kThreads, shared, stream>>>(
          config, start_position, token_count, query, query_norm_scale,
          output_gate, committed.key, committed.value, candidate_rows.key,
          candidate_rows.value, output, normalized_query, tile_spans,
          tile_span_count, tile_span_capacity);
  return cudaPeekAtLastError();
}

cudaError_t launch_pack_committed_kv(
    const AttentionConfig& config, std::size_t token_begin,
    std::size_t token_count, const __nv_bfloat16* physical,
    __nv_bfloat16* logical, cudaStream_t stream) noexcept {
  const std::size_t row_values = attention_kv_row_values(config);
  if (row_values == 0 || token_count == 0 ||
      token_begin >= config.capacity ||
      token_count > config.capacity - token_begin || physical == nullptr ||
      logical == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::size_t values = token_count * row_values;
  const unsigned blocks = static_cast<unsigned>(
      (values + static_cast<std::size_t>(kThreads) - 1) / kThreads);
  pack_committed_kv_kernel<<<blocks == 0 ? 1 : blocks, kThreads, 0, stream>>>(
      physical, logical, token_begin, token_count, config.kv_heads,
      config.capacity, config.head_width);
  return cudaPeekAtLastError();
}

cudaError_t launch_unpack_committed_kv(
    const AttentionConfig& config, std::size_t token_begin,
    std::size_t token_count, const __nv_bfloat16* logical,
    __nv_bfloat16* physical, cudaStream_t stream) noexcept {
  const std::size_t row_values = attention_kv_row_values(config);
  if (row_values == 0 || token_count == 0 ||
      token_begin >= config.capacity ||
      token_count > config.capacity - token_begin || physical == nullptr ||
      logical == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::size_t values = token_count * row_values;
  const unsigned blocks = static_cast<unsigned>(
      (values + static_cast<std::size_t>(kThreads) - 1) / kThreads);
  unpack_committed_kv_kernel<<<blocks == 0 ? 1 : blocks, kThreads, 0, stream>>>(
      logical, physical, token_begin, token_count, config.kv_heads,
      config.capacity, config.head_width);
  return cudaPeekAtLastError();
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

cudaError_t launch_attention_scatter_layers(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, std::size_t layer_count,
    const __nv_bfloat16* candidate_key_base,
    const __nv_bfloat16* candidate_value_base,
    std::size_t candidate_layer_stride, __nv_bfloat16* committed_key_base,
    __nv_bfloat16* committed_value_base, std::size_t committed_layer_stride,
    cudaStream_t stream) noexcept {
  const std::size_t row_values = attention_kv_row_values(config);
  if (row_values == 0 || token_count == 0 || layer_count == 0 ||
      start_position >= config.capacity ||
      token_count > config.capacity - start_position ||
      candidate_key_base == nullptr || candidate_value_base == nullptr ||
      committed_key_base == nullptr || committed_value_base == nullptr ||
      candidate_key_base == committed_key_base ||
      candidate_value_base == committed_value_base) {
    return cudaErrorInvalidValue;
  }
  const unsigned blocks = scatter_block_count(token_count * row_values);
  const dim3 grid(blocks, static_cast<unsigned>(layer_count));
  scatter_committed_rows<<<grid, kThreads, 0, stream>>>(
      candidate_key_base, candidate_value_base, committed_key_base,
      committed_value_base, candidate_layer_stride, committed_layer_stride,
      start_position, token_count, config.kv_heads, config.capacity,
      config.head_width, 0, nullptr);
  return cudaPeekAtLastError();
}

cudaError_t launch_attention_scatter_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const AttentionCache& candidate_rows,
    const AttentionCache& committed, cudaStream_t stream) noexcept {
  return launch_attention_scatter_layers(
      config, start_position, token_count, 1, candidate_rows.key,
      candidate_rows.value, 0, committed.key, committed.value, 0, stream);
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
  const unsigned blocks = scatter_block_count(token_count * row_values);
  scatter_committed_rows<<<dim3(blocks, 1), kThreads, 0, stream>>>(
      candidate_rows.key, candidate_rows.value, committed.key, committed.value,
      0, 0, start_position, token_count, config.kv_heads, config.capacity,
      config.head_width, new_frontier, committed_frontier);
  return cudaPeekAtLastError();
}

}  // namespace qw38::cuda
