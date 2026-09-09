#include "attention_decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_DECODE_KV_AB_RESULT=";
constexpr int kCandidates[] = {1, 4, 8, 16};
constexpr int kCandidateCount = 4;
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr std::size_t kPositions[] = {128, 2048};
constexpr int kPositionCount = 2;
constexpr qw38::cuda::AttentionConfig kConfig{24, 4, 256, 64, 131072};

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

void print_array(FILE* file, const char* key, const std::vector<float>& values) {
  std::fprintf(file, "\"%s\":[", key);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::fprintf(file, ",");
    std::fprintf(file, "%.9g", static_cast<double>(values[index]));
  }
  std::fprintf(file, "]");
}

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  std::size_t nonfinite = 0;
};

Envelope compare(const std::vector<float>& actual,
                 const std::vector<float>& expected) {
  Envelope metrics;
  double squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) ++metrics.nonfinite;
    const float error = std::fabs(actual[index] - expected[index]);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
  }
  metrics.rms = static_cast<float>(
      std::sqrt(squared / static_cast<double>(actual.size())));
  return metrics;
}

cudaError_t launch_candidate(
    int n_parts, const qw38::cuda::AttentionConfig& config,
    std::size_t position, float* query, float* key, float* value,
    float* query_scale, float* key_scale, float* gate,
    const qw38::cuda::AttentionCache& committed,
    const qw38::cuda::AttentionCache& candidate, float* normalized_query,
    float* normalized_key, float* scores, float* output, float* partial,
    float* meta) {
  if (n_parts == 1) {
    return qw38::cuda::launch_attention_prepare(
        config, position, query, key, value, query_scale, key_scale, gate,
        committed, candidate, normalized_query, normalized_key, scores,
        output, nullptr);
  }
  return qw38::cuda::launch_attention_prepare_partitioned(
      config, position, query, key, value, query_scale, key_scale, gate,
      committed, candidate, normalized_query, normalized_key, scores, output,
      partial, meta, n_parts, nullptr);
}

int fill_committed(__nv_bfloat16* device_key, __nv_bfloat16* device_value,
                  std::size_t max_position) {
  const std::size_t cache_values =
      qw38::cuda::attention_cache_values(kConfig);
  cudaError_t error =
      cudaMemset(device_key, 0, cache_values * sizeof(__nv_bfloat16));
  if (error == cudaSuccess) {
    error = cudaMemset(device_value, 0, cache_values * sizeof(__nv_bfloat16));
  }
  if (error != cudaSuccess) return fail_cuda("committed memset", error);
  const std::size_t width = kConfig.head_width;
  std::vector<__nv_bfloat16> prefix_key((max_position + 1) * width);
  std::vector<__nv_bfloat16> prefix_value((max_position + 1) * width);
  for (std::uint32_t kv_head = 0; kv_head < kConfig.kv_heads; ++kv_head) {
    for (std::size_t token = 0; token <= max_position; ++token) {
      for (std::size_t lane = 0; lane < width; ++lane) {
        const std::size_t index = token * width + lane;
        const float key_item =
            static_cast<float>(
                static_cast<int>((lane + token * 5 + kv_head * 11) % 31) -
                15) *
            0.015625F;
        const float value_item =
            static_cast<float>(
                static_cast<int>((lane + token * 7 + kv_head * 13) % 37) -
                18) *
            0.015625F;
        prefix_key[index] = __float2bfloat16_rn(key_item);
        prefix_value[index] = __float2bfloat16_rn(value_item);
      }
    }
    const std::size_t offset = qw38::cuda::attention_kv_physical_index(
        0, kv_head, 0, kConfig.capacity, kConfig.head_width);
    error = cudaMemcpy(device_key + offset, prefix_key.data(),
                       prefix_key.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_value + offset, prefix_value.data(),
                         prefix_value.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) return fail_cuda("committed prefix", error);
  }
  return 0;
}

int run_regime(FILE* raw, std::size_t position, FILE* json) {
  const std::size_t query_values = qw38::cuda::attention_query_values(kConfig);
  const std::size_t row_values = qw38::cuda::attention_kv_row_values(kConfig);
  const std::size_t cache_values = qw38::cuda::attention_cache_values(kConfig);
  const std::size_t score_values =
      qw38::cuda::attention_score_values(kConfig, position);
  const std::size_t vkq_values = qw38::cuda::decode_kv_partial_vkq_values(16);
  const std::size_t meta_values = qw38::cuda::decode_kv_partial_meta_values(16);
  std::vector<float> query(query_values);
  std::vector<float> key(row_values);
  std::vector<float> value(row_values);
  std::vector<float> gate(query_values);
  std::vector<float> query_scale(kConfig.head_width);
  std::vector<float> key_scale(kConfig.head_width);
  std::vector<float> sentinel(score_values, 0.125F);
  for (std::size_t index = 0; index < query_values; ++index) {
    query[index] =
        std::sin(static_cast<float>(index + position) * 0.013F) * 0.5F;
    gate[index] =
        static_cast<float>(static_cast<int>((index + position) % 19) - 9) *
        0.05F;
  }
  for (std::size_t index = 0; index < row_values; ++index) {
    key[index] = std::cos(static_cast<float>(index + position) * 0.019F);
    value[index] =
        static_cast<float>(static_cast<int>((index + position) % 23) - 11) *
        0.04F;
  }
  for (std::uint32_t lane = 0; lane < kConfig.head_width; ++lane) {
    query_scale[lane] = 0.9F + static_cast<float>(lane % 5) * 0.02F;
    key_scale[lane] = 0.85F + static_cast<float>(lane % 6) * 0.02F;
  }

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
  float* device_tiled = nullptr;
  float* device_reference = nullptr;
  float* device_partial = nullptr;
  float* device_meta = nullptr;
  __nv_bfloat16* device_committed_key = nullptr;
  __nv_bfloat16* device_committed_value = nullptr;
  __nv_bfloat16* device_candidate_key = nullptr;
  __nv_bfloat16* device_candidate_value = nullptr;
  cudaError_t error = cudaMalloc(&device_query, query_values * sizeof(float));
#define QW38_ALLOC(pointer, count)                                            \
  if (error == cudaSuccess)                                                \
  error = cudaMalloc(&(pointer), (count) * sizeof(*(pointer)))
  QW38_ALLOC(device_key, row_values);
  QW38_ALLOC(device_value, row_values);
  QW38_ALLOC(device_gate, query_values);
  QW38_ALLOC(device_query_scale, kConfig.head_width);
  QW38_ALLOC(device_key_scale, kConfig.head_width);
  QW38_ALLOC(device_normalized_query, query_values);
  QW38_ALLOC(device_normalized_key, row_values);
  QW38_ALLOC(device_scores, score_values);
  QW38_ALLOC(device_output, query_values);
  QW38_ALLOC(device_tiled, query_values);
  QW38_ALLOC(device_reference, query_values);
  QW38_ALLOC(device_partial, vkq_values);
  QW38_ALLOC(device_meta, meta_values);
  QW38_ALLOC(device_committed_key, cache_values);
  QW38_ALLOC(device_committed_value, cache_values);
  QW38_ALLOC(device_candidate_key, row_values);
  QW38_ALLOC(device_candidate_value, row_values);
#undef QW38_ALLOC
  if (error != cudaSuccess) return fail_cuda("ab cudaMalloc", error);
#define QW38_COPY(pointer, source)                                         \
  if (error == cudaSuccess)                                                \
  error = cudaMemcpy((pointer), (source).data(),                          \
                     (source).size() * sizeof((source)[0]),               \
                     cudaMemcpyHostToDevice)
  QW38_COPY(device_query, query);
  QW38_COPY(device_key, key);
  QW38_COPY(device_value, value);
  QW38_COPY(device_gate, gate);
  QW38_COPY(device_query_scale, query_scale);
  QW38_COPY(device_key_scale, key_scale);
  QW38_COPY(device_scores, sentinel);
#undef QW38_COPY
  if (error != cudaSuccess) return fail_cuda("ab H2D", error);
  if (fill_committed(device_committed_key, device_committed_value,
                      position) != 0) {
    return 1;
  }
  const std::size_t prefix_per_head = (position + 1) * kConfig.head_width;
  std::vector<__nv_bfloat16> original_key(kConfig.kv_heads * prefix_per_head);
  std::vector<__nv_bfloat16> original_value(kConfig.kv_heads * prefix_per_head);
  for (std::uint32_t kv_head = 0; kv_head < kConfig.kv_heads; ++kv_head) {
    const std::size_t offset = qw38::cuda::attention_kv_physical_index(
        0, kv_head, 0, kConfig.capacity, kConfig.head_width);
    error = cudaMemcpy(
        original_key.data() + kv_head * prefix_per_head,
        device_committed_key + offset,
        prefix_per_head * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
      error = cudaMemcpy(original_value.data() + kv_head * prefix_per_head,
                         device_committed_value + offset,
                         prefix_per_head * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) return fail_cuda("ab snapshot", error);
  }
  const qw38::cuda::AttentionCache committed{device_committed_key,
                                            device_committed_value};
  const qw38::cuda::AttentionCache candidate{device_candidate_key,
                                            device_candidate_value};
  error = qw38::cuda::launch_attention_prepare(
      kConfig, position, device_query, device_key, device_value,
      device_query_scale, device_key_scale, device_gate, committed, candidate,
      device_normalized_query, device_normalized_key, device_scores,
      device_tiled, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("ab tiled", error);
  std::vector<__nv_bfloat16> tiled_candidate_key(row_values);
  std::vector<__nv_bfloat16> tiled_candidate_value(row_values);
  error = cudaMemcpy(tiled_candidate_key.data(), device_candidate_key,
                      row_values * sizeof(__nv_bfloat16),
                      cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(tiled_candidate_value.data(), device_candidate_value,
                       row_values * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  error = cudaMemcpy(device_scores, sentinel.data(),
                      score_values * sizeof(float), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_attention_prepare_chunk_reference(
        kConfig, position, 1, device_query, device_key, device_value,
        device_query_scale, device_key_scale, device_gate, committed, candidate,
        device_normalized_query, device_normalized_key, device_scores,
        device_reference, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("ab reference", error);
  std::vector<float> tiled(query_values);
  std::vector<float> reference(query_values);
  error = cudaMemcpy(tiled.data(), device_tiled, query_values * sizeof(float),
                      cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(reference.data(), device_reference,
                      query_values * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("ab baseline D2H", error);

  struct CandidateResult {
    int n_parts = 1;
    int occupancy = 0;
    int merge_occupancy = 0;
    Envelope vs_tiled;
    Envelope vs_reference;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool launch_ok = false;
    bool eligible = false;
    std::vector<float> warmup_ms;
    std::vector<float> samples;
    float mean_ms = 0.0F;
  };
  CandidateResult results[kCandidateCount];

  for (int c = 0; c < kCandidateCount; ++c) {
    const int n_parts = kCandidates[c];
    results[c].n_parts = n_parts;
    results[c].occupancy = qw38::cuda::decode_kv_partition_occupancy(n_parts);
    results[c].merge_occupancy = qw38::cuda::decode_kv_merge_occupancy(n_parts);
    error = cudaMemcpy(device_scores, sentinel.data(),
                        score_values * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return fail_cuda("ab restore", error);
    error = launch_candidate(
        n_parts, kConfig, position, device_query, device_key, device_value,
        device_query_scale, device_key_scale, device_gate, committed,
        candidate, device_normalized_query, device_normalized_key, device_scores,
        device_output, device_partial, device_meta);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    results[c].launch_ok = error == cudaSuccess;
    if (error != cudaSuccess) return fail_cuda("ab candidate launch", error);
    std::vector<float> actual(query_values);
    std::vector<float> scores(score_values);
    std::vector<__nv_bfloat16> actual_candidate_key(row_values);
    std::vector<__nv_bfloat16> actual_candidate_value(row_values);
    error = cudaMemcpy(actual.data(), device_output, query_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
      error = cudaMemcpy(scores.data(), device_scores,
                         score_values * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(actual_candidate_key.data(), device_candidate_key,
                         row_values * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(actual_candidate_value.data(), device_candidate_value,
                         row_values * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) return fail_cuda("ab candidate D2H", error);
    bool committed_unchanged = true;
    for (std::uint32_t kv_head = 0; kv_head < kConfig.kv_heads; ++kv_head) {
      std::vector<__nv_bfloat16> prefix(prefix_per_head);
      const std::size_t offset = qw38::cuda::attention_kv_physical_index(
          0, kv_head, 0, kConfig.capacity, kConfig.head_width);
      error = cudaMemcpy(prefix.data(), device_committed_key + offset,
                         prefix_per_head * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) return fail_cuda("ab committed check", error);
      committed_unchanged =
          committed_unchanged &&
          std::memcmp(prefix.data(),
                      original_key.data() + kv_head * prefix_per_head,
                      prefix_per_head * sizeof(__nv_bfloat16)) == 0;
      error = cudaMemcpy(prefix.data(), device_committed_value + offset,
                         prefix_per_head * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) return fail_cuda("ab committed check", error);
      committed_unchanged =
          committed_unchanged &&
          std::memcmp(prefix.data(),
                      original_value.data() + kv_head * prefix_per_head,
                      prefix_per_head * sizeof(__nv_bfloat16)) == 0;
    }
    results[c].vs_tiled = compare(actual, tiled);
    results[c].vs_reference = compare(actual, reference);
    results[c].scratch_unchanged =
        n_parts == 1 ||
        std::memcmp(scores.data(), sentinel.data(),
                    score_values * sizeof(float)) == 0;
    results[c].committed_unchanged = committed_unchanged;
    results[c].candidate_exact =
        std::memcmp(actual_candidate_key.data(), tiled_candidate_key.data(),
                    row_values * sizeof(__nv_bfloat16)) == 0 &&
        std::memcmp(actual_candidate_value.data(), tiled_candidate_value.data(),
                    row_values * sizeof(__nv_bfloat16)) == 0;
    results[c].eligible =
        results[c].launch_ok && results[c].occupancy >= 1 &&
        results[c].merge_occupancy >= 1 && results[c].vs_tiled.nonfinite == 0 &&
        results[c].vs_reference.nonfinite == 0 &&
        results[c].vs_tiled.max_abs <= 5.0e-5F &&
        results[c].vs_tiled.rms <= 5.0e-6F &&
        results[c].vs_reference.max_abs <= 5.0e-5F &&
        results[c].vs_reference.rms <= 5.0e-6F &&
        results[c].scratch_unchanged && results[c].committed_unchanged &&
        results[c].candidate_exact;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("ab events", error);
  for (int round = 0; round < kWarmups + kMeasured; ++round) {
    for (int c = 0; c < kCandidateCount; ++c) {
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = launch_candidate(
            kCandidates[c], kConfig, position, device_query, device_key,
            device_value, device_query_scale, device_key_scale, device_gate,
            committed, candidate, device_normalized_query,
            device_normalized_key, device_scores, device_output,
            device_partial, device_meta);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float ms = 0.0F;
      if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
      if (error != cudaSuccess) return fail_cuda("ab timed launch", error);
      if (round < kWarmups) {
        results[c].warmup_ms.push_back(ms);
        std::fprintf(raw,
                     "kv_ab position=%zu id=%d warmup=%d ms=%.9g occupancy=%d\n",
                     position, kCandidates[c], round, static_cast<double>(ms),
                     results[c].occupancy);
      } else {
        results[c].samples.push_back(ms);
        std::fprintf(raw,
                     "kv_ab position=%zu id=%d sample=%d ms=%.9g occupancy=%d\n",
                     position, kCandidates[c], round - kWarmups,
                     static_cast<double>(ms), results[c].occupancy);
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  int winner = 1;
  float winner_ms = 0.0F;
  bool have_winner = false;
  for (int c = 0; c < kCandidateCount; ++c) {
    double sum = 0.0;
    for (float sample : results[c].samples) sum += sample;
    results[c].mean_ms =
        static_cast<float>(sum / static_cast<double>(results[c].samples.size()));
    std::fprintf(raw,
                 "kv_ab_mean position=%zu id=%d mean_ms=%.9g occupancy=%d "
                 "eligible=%s\n",
                 position, kCandidates[c],
                 static_cast<double>(results[c].mean_ms),
                 results[c].occupancy, json_bool(results[c].eligible));
    if (!results[c].eligible) continue;
    if (!have_winner || results[c].mean_ms < winner_ms) {
      winner = kCandidates[c];
      winner_ms = results[c].mean_ms;
      have_winner = true;
    }
  }
  const float one_mean = results[0].mean_ms;
  if (!results[0].eligible || winner == 1 || !(winner_ms < one_mean)) {
    winner = 1;
  }
  const bool win = winner != 1;
  std::fprintf(raw,
               "kv_ab_winner position=%zu id=%d win=%s one_ms=%.9g "
               "winner_ms=%.9g\n",
               position, winner, json_bool(win), static_cast<double>(one_mean),
               static_cast<double>(winner == 1 ? one_mean : winner_ms));

    const char* regime = position >= 2048 ? "ab_d2048" : "ab_d128";
  std::fprintf(json, "\"%s\":{\"position\":%zu,\"winner\":%d,\"win\":%s,",
               regime,
               position, winner, json_bool(win));
  std::fprintf(json, "\"candidates\":{");
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::fprintf(json, ",");
    const CandidateResult& result = results[c];
    std::fprintf(json, "\"%d\":{\"n_parts\":%d,\"mean_ms\":%.9g,",
                 result.n_parts, result.n_parts,
                 static_cast<double>(result.mean_ms));
    print_array(json, "samples", result.samples);
    std::fprintf(json, ",");
    print_array(json, "warmup_ms", result.warmup_ms);
    std::fprintf(json,
                 ",\"occupancy\":%d,\"merge_occupancy\":%d,\"launch_ok\":%s,"
                 "\"eligible\":%s,\"scratch_unchanged\":%s,"
                 "\"committed_unchanged\":%s,\"candidate_exact\":%s,"
                 "\"vs_tiled\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu},"
                 "\"vs_reference\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                 "\"nonfinite\":%zu}}",
                 result.occupancy, result.merge_occupancy,
                 json_bool(result.launch_ok), json_bool(result.eligible),
                 json_bool(result.scratch_unchanged),
                 json_bool(result.committed_unchanged),
                 json_bool(result.candidate_exact),
                 static_cast<double>(result.vs_tiled.max_abs),
                 static_cast<double>(result.vs_tiled.rms),
                 result.vs_tiled.nonfinite,
                 static_cast<double>(result.vs_reference.max_abs),
                 static_cast<double>(result.vs_reference.rms),
                 result.vs_reference.nonfinite);
  }
  std::fprintf(json, "}}");

  cudaFree(device_candidate_value);
  cudaFree(device_candidate_key);
  cudaFree(device_committed_value);
  cudaFree(device_committed_key);
  cudaFree(device_meta);
  cudaFree(device_partial);
  cudaFree(device_reference);
  cudaFree(device_tiled);
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
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt036-decode-kv-partition/kv-ab-raw.txt";
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }
  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-036\",\"status\":\"passed\",",
              kPrefix);
  std::printf("\"device\":\"%s\",\"compute_capability\":\"%d.%d\",", prop.name,
              prop.major, prop.minor);
  std::printf("\"measurement_utc\":\"%s\",", utc);
  FILE* json_file = stdout;
  for (int index = 0; index < kPositionCount; ++index) {
    if (index != 0) std::printf(",");
    if (run_regime(raw, kPositions[index], json_file) != 0) {
      std::fclose(raw);
      return 1;
    }
  }
  std::printf(",\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\"}\n");
  std::printf("status=passed\n");
  std::fclose(raw);
  return 0;
}
