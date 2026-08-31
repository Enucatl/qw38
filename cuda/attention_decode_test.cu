#include "attention_decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

float bf16_float(__nv_bfloat16 value) { return __bfloat162float(value); }

void normalize_rope(const float* input, const float* scale,
                    std::uint32_t heads, std::uint32_t width,
                    std::uint32_t rotary_width, std::size_t position,
                    std::vector<float>* output) {
  output->resize(static_cast<std::size_t>(heads) * width);
  for (std::uint32_t head = 0; head < heads; ++head) {
    const std::size_t base = static_cast<std::size_t>(head) * width;
    float sum = 0.0F;
    for (std::uint32_t lane = 0; lane < width; ++lane) {
      sum += input[base + lane] * input[base + lane];
    }
    const float inverse =
        1.0F / std::sqrt(sum / static_cast<float>(width) + 1.0e-6F);
    for (std::uint32_t lane = 0; lane < width; ++lane) {
      (*output)[base + lane] = input[base + lane] * inverse * scale[lane];
    }
    const std::uint32_t half = rotary_width / 2;
    std::vector<float> original(output->begin() + base,
                                output->begin() + base + rotary_width);
    for (std::uint32_t lane = 0; lane < half; ++lane) {
      const float exponent =
          static_cast<float>(lane * 2) / static_cast<float>(rotary_width);
      const float angle = static_cast<float>(position) /
                          std::pow(10000000.0F, exponent);
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      (*output)[base + lane] =
          original[lane] * cosine - original[half + lane] * sine;
      (*output)[base + half + lane] =
          original[half + lane] * cosine + original[lane] * sine;
    }
  }
}

void host_reference(
    const qw38::cuda::AttentionConfig& config, std::size_t position,
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, const std::vector<float>& query_scale,
    const std::vector<float>& key_scale, const std::vector<float>& gate,
    const std::vector<__nv_bfloat16>& committed_key,
    const std::vector<__nv_bfloat16>& committed_value,
    std::vector<float>* normalized_query, std::vector<float>* normalized_key,
    std::vector<__nv_bfloat16>* candidate_key,
    std::vector<__nv_bfloat16>* candidate_value, std::vector<float>* output,
    bool two_byte_current) {
  normalize_rope(query.data(), query_scale.data(), config.query_heads,
                 config.head_width, config.rotary_width, position,
                 normalized_query);
  normalize_rope(key.data(), key_scale.data(), config.kv_heads,
                 config.head_width, config.rotary_width, position,
                 normalized_key);
  const std::size_t query_values =
      qw38::cuda::attention_query_values(config);
  const std::size_t row_values =
      qw38::cuda::attention_kv_row_values(config);
  candidate_key->resize(row_values);
  candidate_value->resize(row_values);
  for (std::size_t index = 0; index < row_values; ++index) {
    (*candidate_key)[index] = __float2bfloat16_rn((*normalized_key)[index]);
    (*candidate_value)[index] = __float2bfloat16_rn(value[index]);
  }
  output->assign(query_values, 0.0F);
  const std::uint32_t group_size = config.query_heads / config.kv_heads;
  const float scaling =
      1.0F / std::sqrt(static_cast<float>(config.head_width));
  std::vector<float> scores(position + 1);
  for (std::uint32_t query_head = 0; query_head < config.query_heads;
       ++query_head) {
    const std::uint32_t kv_head = query_head / group_size;
    const std::size_t query_base =
        static_cast<std::size_t>(query_head) * config.head_width;
    const std::size_t kv_base =
        static_cast<std::size_t>(kv_head) * config.head_width;
    float maximum = -INFINITY;
    for (std::size_t context = 0; context <= position; ++context) {
      float score = 0.0F;
      for (std::uint32_t lane = 0; lane < config.head_width; ++lane) {
        const float key_item =
            context == position && !two_byte_current
                ? (*normalized_key)[kv_base + lane]
                : bf16_float((context == position
                                  ? candidate_key->data()
                                  : committed_key.data() + context * row_values)
                                 [kv_base + lane]);
        score += (*normalized_query)[query_base + lane] *
                 key_item;
      }
      scores[context] = score * scaling;
      maximum = std::max(maximum, scores[context]);
    }
    float denominator = 0.0F;
    for (std::size_t context = 0; context <= position; ++context) {
      scores[context] = std::exp(scores[context] - maximum);
      denominator += scores[context];
    }
    for (std::uint32_t lane = 0; lane < config.head_width; ++lane) {
      float result = 0.0F;
      for (std::size_t context = 0; context <= position; ++context) {
        const float value_item =
            context == position && !two_byte_current
                ? value[kv_base + lane]
                : bf16_float((context == position
                                  ? candidate_value->data()
                                  : committed_value.data() +
                                        context * row_values)[kv_base + lane]);
        result += scores[context] / denominator *
                  value_item;
      }
      const float gate_value = gate[query_base + lane];
      const float sigmoid = gate_value >= 0.0F
                                ? 1.0F / (1.0F + std::exp(-gate_value))
                                : std::exp(gate_value) /
                                      (1.0F + std::exp(gate_value));
      (*output)[query_base + lane] = result * sigmoid;
    }
  }
}

struct Metrics {
  float maximum_absolute = 0.0F;
  double squared = 0.0;
  std::size_t count = 0;
  std::size_t nonfinite = 0;
};

void add_metrics(const std::vector<float>& actual,
                 const std::vector<float>& expected, Metrics* metrics) {
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) ++metrics->nonfinite;
    const float error = std::fabs(actual[index] - expected[index]);
    metrics->maximum_absolute = std::max(metrics->maximum_absolute, error);
    metrics->squared += static_cast<double>(error) * error;
    ++metrics->count;
  }
}

double cosine_similarity(const std::vector<float>& actual,
                         const std::vector<float>& expected) {
  double dot = 0.0;
  double actual_squared = 0.0;
  double expected_squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    dot += static_cast<double>(actual[index]) * expected[index];
    actual_squared += static_cast<double>(actual[index]) * actual[index];
    expected_squared += static_cast<double>(expected[index]) * expected[index];
  }
  return dot / std::sqrt(actual_squared * expected_squared);
}

bool bf16_equal(const std::vector<__nv_bfloat16>& left,
                const std::vector<__nv_bfloat16>& right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(),
                     left.size() * sizeof(left[0])) == 0;
}

int run_case(const char* name, std::uint32_t layer,
             const qw38::cuda::AttentionConfig& config) {
  const std::size_t position = 3;
  const std::size_t query_values =
      qw38::cuda::attention_query_values(config);
  const std::size_t row_values =
      qw38::cuda::attention_kv_row_values(config);
  const std::size_t cache_values =
      qw38::cuda::attention_cache_values(config);
  const std::size_t score_values =
      qw38::cuda::attention_score_values(config, position);
  std::vector<float> query(query_values);
  std::vector<float> key(row_values);
  std::vector<float> value(row_values);
  std::vector<float> gate(query_values);
  std::vector<float> query_scale(config.head_width);
  std::vector<float> key_scale(config.head_width);
  for (std::size_t index = 0; index < query_values; ++index) {
    query[index] =
        std::sin(static_cast<float>(index + layer) * 0.017F) * 0.75F;
    gate[index] =
        static_cast<float>(static_cast<int>((index + layer) % 17) - 8) *
        0.0625F;
  }
  for (std::size_t index = 0; index < row_values; ++index) {
    key[index] = std::cos(static_cast<float>(index + layer * 3) * 0.023F);
    value[index] =
        static_cast<float>(static_cast<int>((index + layer) % 29) - 14) *
        0.03125F;
  }
  for (std::size_t lane = 0; lane < config.head_width; ++lane) {
    query_scale[lane] = 0.875F + static_cast<float>(lane % 9) * 0.03125F;
    key_scale[lane] = 0.9375F + static_cast<float>(lane % 7) * 0.015625F;
  }
  std::vector<__nv_bfloat16> committed_key(cache_values);
  std::vector<__nv_bfloat16> committed_value(cache_values);
  for (std::size_t context = 0; context < config.capacity; ++context) {
    for (std::size_t index = 0; index < row_values; ++index) {
      const float key_item =
          context > position
              ? 1000.0F
              : static_cast<float>(static_cast<int>((index + context * 5) % 31) -
                                   15) *
                    0.015625F;
      const float value_item =
          context > position
              ? -1000.0F
              : static_cast<float>(static_cast<int>((index + context * 7) % 37) -
                                   18) *
                    0.015625F;
      committed_key[context * row_values + index] =
          __float2bfloat16_rn(key_item);
      committed_value[context * row_values + index] =
          __float2bfloat16_rn(value_item);
    }
  }
  const std::vector<__nv_bfloat16> original_key = committed_key;
  const std::vector<__nv_bfloat16> original_value = committed_value;
  std::vector<float> expected_query;
  std::vector<float> expected_key;
  std::vector<__nv_bfloat16> expected_candidate_key;
  std::vector<__nv_bfloat16> expected_candidate_value;
  std::vector<float> expected_output;
  host_reference(config, position, query, key, value, query_scale, key_scale,
                 gate, committed_key, committed_value, &expected_query,
                 &expected_key, &expected_candidate_key,
                 &expected_candidate_value, &expected_output, true);
  std::vector<float> scalar_query;
  std::vector<float> scalar_key;
  std::vector<__nv_bfloat16> scalar_candidate_key;
  std::vector<__nv_bfloat16> scalar_candidate_value;
  std::vector<float> scalar_output;
  host_reference(config, position, query, key, value, query_scale, key_scale,
                 gate, committed_key, committed_value, &scalar_query,
                 &scalar_key, &scalar_candidate_key, &scalar_candidate_value,
                 &scalar_output, false);

  float* device_query = nullptr;
  float* device_key = nullptr;
  float* device_value = nullptr;
  float* device_gate = nullptr;
  float* device_query_scale = nullptr;
  float* device_key_scale = nullptr;
  float* device_normalized_query = nullptr;
  float* device_normalized_key = nullptr;
  float* device_scores = nullptr;
  float* device_output = nullptr;
  __nv_bfloat16* device_committed_key = nullptr;
  __nv_bfloat16* device_committed_value = nullptr;
  __nv_bfloat16* device_candidate_key = nullptr;
  __nv_bfloat16* device_candidate_value = nullptr;
  std::uint64_t* device_frontier = nullptr;
  cudaError_t error = cudaMalloc(&device_query, query_values * sizeof(float));
#define QW38_ALLOC(pointer, count)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMalloc(&(pointer), (count) * sizeof(*(pointer)))
  QW38_ALLOC(device_key, row_values);
  QW38_ALLOC(device_value, row_values);
  QW38_ALLOC(device_gate, query_values);
  QW38_ALLOC(device_query_scale, config.head_width);
  QW38_ALLOC(device_key_scale, config.head_width);
  QW38_ALLOC(device_normalized_query, query_values);
  QW38_ALLOC(device_normalized_key, row_values);
  QW38_ALLOC(device_scores, score_values);
  QW38_ALLOC(device_output, query_values);
  QW38_ALLOC(device_committed_key, cache_values);
  QW38_ALLOC(device_committed_value, cache_values);
  QW38_ALLOC(device_candidate_key, row_values);
  QW38_ALLOC(device_candidate_value, row_values);
  QW38_ALLOC(device_frontier, 1);
#undef QW38_ALLOC
  if (error != cudaSuccess) return fail_cuda("attention cudaMalloc", error);
#define QW38_COPY(pointer, source)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((pointer), (source).data(),                              \
                     (source).size() * sizeof((source)[0]), cudaMemcpyHostToDevice)
  QW38_COPY(device_query, query);
  QW38_COPY(device_key, key);
  QW38_COPY(device_value, value);
  QW38_COPY(device_gate, gate);
  QW38_COPY(device_query_scale, query_scale);
  QW38_COPY(device_key_scale, key_scale);
  QW38_COPY(device_committed_key, committed_key);
  QW38_COPY(device_committed_value, committed_value);
#undef QW38_COPY
  const std::uint64_t initial_frontier = position;
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_frontier, &initial_frontier,
                       sizeof(initial_frontier), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("attention cudaMemcpy H2D", error);
  const qw38::cuda::AttentionCache committed{device_committed_key,
                                              device_committed_value};
  const qw38::cuda::AttentionCache candidate{device_candidate_key,
                                              device_candidate_value};
  if (qw38::cuda::launch_attention_prepare(
          config, position, device_query, device_key, device_value,
          device_query_scale, device_key_scale, device_gate, committed,
          committed, device_normalized_query, device_normalized_key,
          device_scores, device_output, nullptr) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "%s: aliased candidate was not rejected\n", name);
    return 1;
  }
  for (int warmup = 0; warmup < 3 && error == cudaSuccess; ++warmup) {
    error = qw38::cuda::launch_attention_prepare(
        config, position, device_query, device_key, device_value,
        device_query_scale, device_key_scale, device_gate, committed, candidate,
        device_normalized_query, device_normalized_key, device_scores,
        device_output, nullptr);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  for (int sample = 0; sample < 30 && error == cudaSuccess; ++sample) {
    error = qw38::cuda::launch_attention_prepare(
        config, position, device_query, device_key, device_value,
        device_query_scale, device_key_scale, device_gate, committed, candidate,
        device_normalized_query, device_normalized_key, device_scores,
        device_output, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return fail_cuda("attention prepare", error);
  float milliseconds = 0.0F;
  error = cudaEventElapsedTime(&milliseconds, start, stop);
  if (error != cudaSuccess) return fail_cuda("attention timing", error);

  std::vector<float> actual_query(query_values);
  std::vector<float> actual_key(row_values);
  std::vector<float> actual_output(query_values);
  std::vector<__nv_bfloat16> actual_candidate_key(row_values);
  std::vector<__nv_bfloat16> actual_candidate_value(row_values);
#define QW38_READ(destination, pointer)                                       \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((destination).data(), (pointer),                         \
                     (destination).size() * sizeof((destination)[0]),         \
                     cudaMemcpyDeviceToHost)
  QW38_READ(actual_query, device_normalized_query);
  QW38_READ(actual_key, device_normalized_key);
  QW38_READ(actual_output, device_output);
  QW38_READ(actual_candidate_key, device_candidate_key);
  QW38_READ(actual_candidate_value, device_candidate_value);
  QW38_READ(committed_key, device_committed_key);
  QW38_READ(committed_value, device_committed_value);
  std::uint64_t actual_frontier = 0;
  if (error == cudaSuccess) {
    error = cudaMemcpy(&actual_frontier, device_frontier,
                       sizeof(actual_frontier), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("attention cudaMemcpy D2H", error);
  const bool prepare_atomic = bf16_equal(committed_key, original_key) &&
                              bf16_equal(committed_value, original_value) &&
                              actual_frontier == initial_frontier;
  const bool candidate_exact =
      bf16_equal(actual_candidate_key, expected_candidate_key) &&
      bf16_equal(actual_candidate_value, expected_candidate_value);
  Metrics metrics;
  add_metrics(actual_query, expected_query, &metrics);
  add_metrics(actual_key, expected_key, &metrics);
  add_metrics(actual_output, expected_output, &metrics);
  const float rms = static_cast<float>(
      std::sqrt(metrics.squared / static_cast<double>(metrics.count)));
  Metrics oracle_metrics;
  add_metrics(actual_output, scalar_output, &oracle_metrics);
  const float oracle_rms = static_cast<float>(std::sqrt(
      oracle_metrics.squared / static_cast<double>(oracle_metrics.count)));
  const double oracle_cosine = cosine_similarity(actual_output, scalar_output);

  error = qw38::cuda::launch_attention_commit(
      config, position, candidate, committed, position + 1, device_frontier,
      nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("attention commit", error);
  QW38_READ(committed_key, device_committed_key);
  QW38_READ(committed_value, device_committed_value);
#undef QW38_READ
  if (error == cudaSuccess) {
    error = cudaMemcpy(&actual_frontier, device_frontier,
                       sizeof(actual_frontier), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("attention committed read", error);
  bool commit_exact = actual_frontier == position + 1;
  for (std::size_t context = 0; context < config.capacity; ++context) {
    for (std::size_t index = 0; index < row_values; ++index) {
      const std::size_t flat = context * row_values + index;
      const __nv_bfloat16 expected_committed_key =
          context == position ? expected_candidate_key[index]
                              : original_key[flat];
      const __nv_bfloat16 expected_committed_value =
          context == position ? expected_candidate_value[index]
                              : original_value[flat];
      commit_exact =
          commit_exact &&
          std::memcmp(&committed_key[flat], &expected_committed_key,
                      sizeof(expected_committed_key)) == 0 &&
          std::memcmp(&committed_value[flat], &expected_committed_value,
                      sizeof(expected_committed_value)) == 0;
    }
  }
  std::printf(
      "attention_case=%s layer=%u query_heads=%u kv_heads=%u width=%u "
      "rotary=%u max_abs=%.9g rms=%.9g oracle_max_abs=%.9g "
      "oracle_rms=%.9g oracle_cosine=%.12g nonfinite=%zu candidate_exact=%s "
      "prepare_atomic=%s commit_exact=%s mean_ms=%.9g\n",
      name, layer, config.query_heads, config.kv_heads, config.head_width,
      config.rotary_width, metrics.maximum_absolute, rms,
      oracle_metrics.maximum_absolute, oracle_rms, oracle_cosine,
      metrics.nonfinite,
      candidate_exact ? "true" : "false", prepare_atomic ? "true" : "false",
      commit_exact ? "true" : "false", milliseconds / 30.0F);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_frontier);
  cudaFree(device_candidate_value);
  cudaFree(device_candidate_key);
  cudaFree(device_committed_value);
  cudaFree(device_committed_key);
  cudaFree(device_output);
  cudaFree(device_scores);
  cudaFree(device_normalized_key);
  cudaFree(device_normalized_query);
  cudaFree(device_key_scale);
  cudaFree(device_query_scale);
  cudaFree(device_gate);
  cudaFree(device_value);
  cudaFree(device_key);
  cudaFree(device_query);
  return metrics.nonfinite == 0 && metrics.maximum_absolute <= 5.0e-5F &&
                 rms <= 5.0e-6F && oracle_metrics.maximum_absolute <= 0.051F &&
                 oracle_rms <= 0.0016F && oracle_cosine >= 0.999424 &&
                 candidate_exact && prepare_atomic && commit_exact
             ? 0
             : 1;
}

}  // namespace

int main() {
  const qw38::cuda::AttentionConfig invalid{5, 2, 8, 4, 4};
  if (qw38::cuda::attention_query_values(invalid) != 0) {
    std::fprintf(stderr, "invalid attention shape was accepted\n");
    return 1;
  }
  const qw38::cuda::AttentionConfig small{6, 2, 8, 4, 5};
  const qw38::cuda::AttentionConfig production{24, 4, 256, 64, 5};
  if (run_case("layer_3", 3, small) != 0 ||
      run_case("layer_7", 7, small) != 0 ||
      run_case("layer_63", 63, small) != 0 ||
      run_case("production_layer_3", 3, production) != 0) {
    return 1;
  }
  std::printf("status=passed\n");
  return 0;
}
