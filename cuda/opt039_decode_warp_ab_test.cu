#include "attention_decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_DECODE_WARP_AB_RESULT=";
constexpr const char* kCandidates[] = {"cta_group", "warp_query"};
constexpr int kCandidateCount = 2;
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr std::size_t kTimedPositions[] = {128, 2048};
constexpr std::size_t kCorrectnessPositions[] = {
    0, 1, 15, 16, 31, 32, 127, 128, 2047, 2048, 2303, 131071};
constexpr int kCorrectnessCount = 12;
constexpr qw38::cuda::AttentionConfig kConfig{24, 4, 256, 64, 131072};
constexpr int kParts = 16;

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

void print_envelope(FILE* file, const char* key, const Envelope& metrics,
                    bool present) {
  if (!present) {
    std::fprintf(file, "\"%s\":null", key);
    return;
  }
  std::fprintf(file,
               "\"%s\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu}", key,
               static_cast<double>(metrics.max_abs),
               static_cast<double>(metrics.rms), metrics.nonfinite);
}

bool envelope_ok(const Envelope& metrics) {
  return metrics.nonfinite == 0 && metrics.max_abs <= 5.0e-5F &&
         metrics.rms <= 5.0e-6F;
}

cudaError_t launch_vec(
    const char* vec_path, const qw38::cuda::AttentionConfig& config,
    std::size_t position, float* query, float* key, float* value,
    float* query_scale, float* key_scale, float* gate,
    const qw38::cuda::AttentionCache& committed,
    const qw38::cuda::AttentionCache& candidate, float* normalized_query,
    float* normalized_key, float* scores, float* output, float* partial,
    float* meta) {
  return qw38::cuda::launch_attention_prepare_partitioned_vec(
      config, position, query, key, value, query_scale, key_scale, gate,
      committed, candidate, normalized_query, normalized_key, scores, output,
      partial, meta, kParts, vec_path, nullptr);
}

int occupancy_for(const char* vec_path) {
  if (std::strcmp(vec_path, "warp_query") == 0)
    return qw38::cuda::decode_kv_warp_query_occupancy();
  return qw38::cuda::decode_kv_partition_occupancy(kParts);
}

int fill_committed(__nv_bfloat16* device_key, __nv_bfloat16* device_value,
                   std::size_t max_position) {
  const std::size_t cache_values = qw38::cuda::attention_cache_values(kConfig);
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

struct CandidateResult {
  const char* id = "cta_group";
  int occupancy = 0;
  int merge_occupancy = 0;
  Envelope vs_cta;
  Envelope vs_tiled;
  Envelope vs_reference;
  bool skip_tiled_reference = false;
  bool scratch_unchanged = false;
  bool committed_unchanged = false;
  bool candidate_exact = false;
  bool launch_ok = false;
  bool eligible = false;
  std::vector<float> warmup_ms;
  std::vector<float> samples;
  float mean_ms = 0.0F;
};

void print_candidate(FILE* file, const CandidateResult& result, bool timed) {
  std::fprintf(file, "\"%s\":{\"id\":\"%s\",\"mean_ms\":%.9g,", result.id,
               result.id, static_cast<double>(result.mean_ms));
  if (timed) {
    print_array(file, "samples", result.samples);
    std::fprintf(file, ",");
    print_array(file, "warmup_ms", result.warmup_ms);
    std::fprintf(file, ",");
  }
  std::fprintf(file,
               "\"occupancy\":%d,\"merge_occupancy\":%d,\"launch_ok\":%s,"
               "\"eligible\":%s,\"scratch_unchanged\":%s,"
               "\"committed_unchanged\":%s,\"candidate_exact\":%s,"
               "\"skip_tiled_reference\":%s,",
               result.occupancy, result.merge_occupancy,
               json_bool(result.launch_ok), json_bool(result.eligible),
               json_bool(result.scratch_unchanged),
               json_bool(result.committed_unchanged),
               json_bool(result.candidate_exact),
               json_bool(result.skip_tiled_reference));
  print_envelope(file, "vs_cta", result.vs_cta, true);
  std::fprintf(file, ",");
  print_envelope(file, "vs_tiled", result.vs_tiled,
                 !result.skip_tiled_reference);
  std::fprintf(file, ",");
  print_envelope(file, "vs_reference", result.vs_reference,
                 !result.skip_tiled_reference);
  std::fprintf(file, "}");
}

bool committed_matches(__nv_bfloat16* device_key, __nv_bfloat16* device_value,
                       const std::vector<__nv_bfloat16>& original_key,
                       const std::vector<__nv_bfloat16>& original_value,
                       std::size_t prefix_per_head) {
  cudaError_t error = cudaSuccess;
  for (std::uint32_t kv_head = 0; kv_head < kConfig.kv_heads; ++kv_head) {
    std::vector<__nv_bfloat16> prefix(prefix_per_head);
    const std::size_t offset = qw38::cuda::attention_kv_physical_index(
        0, kv_head, 0, kConfig.capacity, kConfig.head_width);
    error = cudaMemcpy(prefix.data(), device_key + offset,
                       prefix_per_head * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return false;
    if (std::memcmp(prefix.data(),
                    original_key.data() + kv_head * prefix_per_head,
                    prefix_per_head * sizeof(__nv_bfloat16)) != 0) {
      return false;
    }
    error = cudaMemcpy(prefix.data(), device_value + offset,
                       prefix_per_head * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return false;
    if (std::memcmp(prefix.data(),
                    original_value.data() + kv_head * prefix_per_head,
                    prefix_per_head * sizeof(__nv_bfloat16)) != 0) {
      return false;
    }
  }
  return true;
}

int evaluate_position(FILE* json, std::size_t position, bool timed, FILE* raw,
                      bool* warp_eligible) {
  const bool skip_refs = position == 131071;
  const std::size_t query_values = qw38::cuda::attention_query_values(kConfig);
  const std::size_t row_values = qw38::cuda::attention_kv_row_values(kConfig);
  const std::size_t cache_values = qw38::cuda::attention_cache_values(kConfig);
  const std::size_t score_values =
      qw38::cuda::attention_score_values(kConfig, position);
  const std::size_t vkq_values = qw38::cuda::decode_kv_partial_vkq_values(kParts);
  const std::size_t meta_values =
      qw38::cuda::decode_kv_partial_meta_values(kParts);
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
  float* device_cta = nullptr;
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
  QW38_ALLOC(device_cta, query_values);
  if (!skip_refs) {
    QW38_ALLOC(device_tiled, query_values);
    QW38_ALLOC(device_reference, query_values);
  }
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
  if (fill_committed(device_committed_key, device_committed_value, position) !=
      0) {
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

  std::vector<float> tiled;
  std::vector<float> reference;
  std::vector<__nv_bfloat16> gold_candidate_key(row_values);
  std::vector<__nv_bfloat16> gold_candidate_value(row_values);
  if (!skip_refs) {
    error = qw38::cuda::launch_attention_prepare(
        kConfig, position, device_query, device_key, device_value,
        device_query_scale, device_key_scale, device_gate, committed, candidate,
        device_normalized_query, device_normalized_key, device_scores,
        device_tiled, nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) return fail_cuda("ab tiled", error);
    error = cudaMemcpy(gold_candidate_key.data(), device_candidate_key,
                       row_values * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
      error = cudaMemcpy(gold_candidate_value.data(), device_candidate_value,
                         row_values * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    }
    error = cudaMemcpy(device_scores, sentinel.data(),
                       score_values * sizeof(float), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_attention_prepare_chunk_reference(
          kConfig, position, 1, device_query, device_key, device_value,
          device_query_scale, device_key_scale, device_gate, committed,
          candidate, device_normalized_query, device_normalized_key,
          device_scores, device_reference, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) return fail_cuda("ab reference", error);
    tiled.assign(query_values, 0.0F);
    reference.assign(query_values, 0.0F);
    error = cudaMemcpy(tiled.data(), device_tiled, query_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
      error = cudaMemcpy(reference.data(), device_reference,
                         query_values * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) return fail_cuda("ab baseline D2H", error);
  }

  CandidateResult results[kCandidateCount];
  std::vector<float> cta_output(query_values);
  for (int c = 0; c < kCandidateCount; ++c) {
    results[c].id = kCandidates[c];
    results[c].occupancy = occupancy_for(kCandidates[c]);
    results[c].merge_occupancy = qw38::cuda::decode_kv_merge_occupancy(kParts);
    results[c].skip_tiled_reference = skip_refs;
    error = cudaMemcpy(device_scores, sentinel.data(),
                       score_values * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return fail_cuda("ab restore", error);
    error = launch_vec(kCandidates[c], kConfig, position, device_query,
                       device_key, device_value, device_query_scale,
                       device_key_scale, device_gate, committed, candidate,
                       device_normalized_query, device_normalized_key,
                       device_scores, device_output, device_partial,
                       device_meta);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    results[c].launch_ok = error == cudaSuccess;
    if (error != cudaSuccess) return fail_cuda("ab candidate launch", error);
    std::vector<float> actual(query_values);
    std::vector<float> scores(score_values);
    std::vector<__nv_bfloat16> actual_candidate_key(row_values);
    std::vector<__nv_bfloat16> actual_candidate_value(row_values);
    error = cudaMemcpy(actual.data(), device_output,
                       query_values * sizeof(float), cudaMemcpyDeviceToHost);
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
    if (c == 0) {
      cta_output = actual;
      if (skip_refs) {
        gold_candidate_key = actual_candidate_key;
        gold_candidate_value = actual_candidate_value;
      }
      error = cudaMemcpy(device_cta, device_output,
                         query_values * sizeof(float), cudaMemcpyDeviceToDevice);
      if (error != cudaSuccess) return fail_cuda("ab cta stash", error);
    }
    results[c].vs_cta = compare(actual, cta_output);
    if (!skip_refs) {
      results[c].vs_tiled = compare(actual, tiled);
      results[c].vs_reference = compare(actual, reference);
    }
    results[c].scratch_unchanged =
        std::memcmp(scores.data(), sentinel.data(),
                    score_values * sizeof(float)) == 0;
    results[c].committed_unchanged = committed_matches(
        device_committed_key, device_committed_value, original_key,
        original_value, prefix_per_head);
    results[c].candidate_exact =
        std::memcmp(actual_candidate_key.data(), gold_candidate_key.data(),
                    row_values * sizeof(__nv_bfloat16)) == 0 &&
        std::memcmp(actual_candidate_value.data(), gold_candidate_value.data(),
                    row_values * sizeof(__nv_bfloat16)) == 0;
    const bool numeric =
        envelope_ok(results[c].vs_cta) &&
        (skip_refs || (envelope_ok(results[c].vs_tiled) &&
                       envelope_ok(results[c].vs_reference)));
    results[c].eligible =
        results[c].launch_ok && results[c].occupancy >= 1 &&
        results[c].merge_occupancy >= 1 && numeric &&
        results[c].scratch_unchanged && results[c].committed_unchanged &&
        results[c].candidate_exact;
  }
  if (!results[0].eligible) {
    std::fprintf(stderr, "cta_group ineligible at position %zu\n", position);
    return 1;
  }

  if (timed) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    if (error != cudaSuccess) return fail_cuda("ab events", error);
    for (int round = 0; round < kWarmups + kMeasured; ++round) {
      for (int c = 0; c < kCandidateCount; ++c) {
        error = cudaEventRecord(start);
        if (error == cudaSuccess) {
          error = launch_vec(
              kCandidates[c], kConfig, position, device_query, device_key,
              device_value, device_query_scale, device_key_scale, device_gate,
              committed, candidate, device_normalized_query,
              device_normalized_key, device_scores, device_output,
              device_partial, device_meta);
        }
        if (error == cudaSuccess) error = cudaEventRecord(stop);
        if (error == cudaSuccess) error = cudaEventSynchronize(stop);
        float ms = 0.0F;
        if (error == cudaSuccess)
          error = cudaEventElapsedTime(&ms, start, stop);
        if (error != cudaSuccess) return fail_cuda("ab timed launch", error);
        if (round < kWarmups) {
          results[c].warmup_ms.push_back(ms);
          std::fprintf(raw,
                       "warp_ab position=%zu id=%s warmup=%d ms=%.9g "
                       "occupancy=%d\n",
                       position, kCandidates[c], round,
                       static_cast<double>(ms), results[c].occupancy);
        } else {
          results[c].samples.push_back(ms);
          std::fprintf(raw,
                       "warp_ab position=%zu id=%s sample=%d ms=%.9g "
                       "occupancy=%d\n",
                       position, kCandidates[c], round - kWarmups,
                       static_cast<double>(ms), results[c].occupancy);
        }
      }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int c = 0; c < kCandidateCount; ++c) {
      double sum = 0.0;
      for (float sample : results[c].samples) sum += sample;
      results[c].mean_ms = static_cast<float>(
          sum / static_cast<double>(results[c].samples.size()));
      std::fprintf(raw,
                   "warp_ab_mean position=%zu id=%s mean_ms=%.9g occupancy=%d "
                   "eligible=%s\n",
                   position, kCandidates[c],
                   static_cast<double>(results[c].mean_ms),
                   results[c].occupancy, json_bool(results[c].eligible));
    }
    const bool warp_win = results[1].eligible && results[0].eligible &&
                          results[1].mean_ms < results[0].mean_ms;
    const char* winner = warp_win ? "warp_query" : "cta_group";
    std::fprintf(raw,
                 "warp_ab_winner position=%zu id=%s win=%s cta_ms=%.9g "
                 "warp_ms=%.9g\n",
                 position, winner, json_bool(warp_win),
                 static_cast<double>(results[0].mean_ms),
                 static_cast<double>(results[1].mean_ms));
    const char* regime = position >= 2048 ? "ab_d2048" : "ab_d128";
    std::fprintf(json, "\"%s\":{\"position\":%zu,\"winner\":\"%s\",\"win\":%s,"
                       "\"candidates\":{",
                 regime, position, winner, json_bool(warp_win));
    for (int c = 0; c < kCandidateCount; ++c) {
      if (c != 0) std::fprintf(json, ",");
      print_candidate(json, results[c], true);
    }
    std::fprintf(json, "}}");
  } else {
    std::fprintf(json, "\"%zu\":{\"position\":%zu,", position, position);
    for (int c = 0; c < kCandidateCount; ++c) {
      if (c != 0) std::fprintf(json, ",");
      print_candidate(json, results[c], false);
    }
    std::fprintf(json, "}");
    std::fprintf(raw,
                 "warp_ab_correctness position=%zu warp_eligible=%s "
                 "cta_eligible=%s vs_cta_max=%.9g vs_cta_rms=%.9g\n",
                 position, json_bool(results[1].eligible),
                 json_bool(results[0].eligible),
                 static_cast<double>(results[1].vs_cta.max_abs),
                 static_cast<double>(results[1].vs_cta.rms));
  }
  if (warp_eligible != nullptr) *warp_eligible = results[1].eligible;

  cudaFree(device_candidate_value);
  cudaFree(device_candidate_key);
  cudaFree(device_committed_value);
  cudaFree(device_committed_key);
  cudaFree(device_meta);
  cudaFree(device_partial);
  if (device_reference != nullptr) cudaFree(device_reference);
  if (device_tiled != nullptr) cudaFree(device_tiled);
  cudaFree(device_cta);
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
               : "evidence/optimization/opt039-decode-warp/warp-ab-raw.txt";
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
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-039\",\"status\":\"passed\",",
              kPrefix);
  std::printf("\"device\":\"%s\",\"compute_capability\":\"%d.%d\",", prop.name,
              prop.major, prop.minor);
  std::printf("\"measurement_utc\":\"%s\",", utc);
  FILE* json_file = stdout;
  std::printf("\"correctness\":{\"positions\":[");
  for (int index = 0; index < kCorrectnessCount; ++index) {
    if (index != 0) std::printf(",");
    std::printf("%zu", kCorrectnessPositions[index]);
  }
  std::printf("],\"cases\":{");
  bool all_correct = true;
  for (int index = 0; index < kCorrectnessCount; ++index) {
    if (index != 0) std::printf(",");
    bool warp_ok = false;
    if (evaluate_position(json_file, kCorrectnessPositions[index], false, raw,
                          &warp_ok) != 0) {
      std::fclose(raw);
      return 1;
    }
    all_correct = all_correct && warp_ok;
  }
  std::printf("},\"all_eligible\":%s},", json_bool(all_correct));
  bool d128_warp = false;
  bool d2048_warp = false;
  if (evaluate_position(json_file, kTimedPositions[0], true, raw, &d128_warp) !=
      0) {
    std::fclose(raw);
    return 1;
  }
  std::printf(",");
  if (evaluate_position(json_file, kTimedPositions[1], true, raw, &d2048_warp) !=
      0) {
    std::fclose(raw);
    return 1;
  }
  std::printf(
      ",\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\"}\n");
  std::printf("status=passed\n");
  std::fflush(stdout);
  std::fclose(raw);
  (void)d128_warp;
  (void)d2048_warp;
  (void)all_correct;
  return 0;
}
