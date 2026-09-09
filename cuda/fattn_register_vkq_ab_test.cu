#include "attention_decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_FATTN_REGISTER_VKQ_AB_RESULT=";
constexpr const char* kCandidates[] = {"global", "registers"};
constexpr int kCandidateCount = 2;
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr std::size_t kTokenCount = 4096;
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

int occupancy_for(const char* vkq_accum) {
  if (std::strcmp(vkq_accum, "registers") == 0)
    return qw38::cuda::fattn_register_vkq_occupancy();
  return qw38::cuda::attention_mma_quality_occupancy_for_path("stream_k");
}

}  // namespace

int main(int argc, char** argv) {
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt033-register-vkq/vkq-ab-raw.txt";
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  const std::size_t query_values =
      kTokenCount * qw38::cuda::attention_query_values(kConfig);
  const std::size_t row_values =
      kTokenCount * qw38::cuda::attention_kv_row_values(kConfig);
  const std::size_t cache_values = qw38::cuda::attention_cache_values(kConfig);
  const std::size_t score_values = qw38::cuda::attention_chunk_score_values(
      kConfig, 0, kTokenCount);
  const std::size_t partial_values =
      qw38::cuda::fattn_stream_k_partial_values(kConfig, kTokenCount);
  const std::size_t meta_values =
      qw38::cuda::fattn_stream_k_meta_values(kConfig, kTokenCount);

  std::vector<float> query(query_values);
  std::vector<float> key(row_values);
  std::vector<float> value(row_values);
  std::vector<float> gate(query_values);
  std::vector<float> query_scale(kConfig.head_width);
  std::vector<float> key_scale(kConfig.head_width);
  std::vector<float> sentinel(score_values, 0.125F);
  for (std::size_t index = 0; index < query_values; ++index) {
    query[index] = std::sin(static_cast<float>(index) * 0.0013F) * 0.5F;
    gate[index] =
        static_cast<float>(static_cast<int>(index % 19) - 9) * 0.05F;
  }
  for (std::size_t index = 0; index < row_values; ++index) {
    key[index] = std::cos(static_cast<float>(index) * 0.0019F);
    value[index] =
        static_cast<float>(static_cast<int>(index % 23) - 11) * 0.04F;
  }
  for (std::uint32_t lane = 0; lane < kConfig.head_width; ++lane) {
    query_scale[lane] = 0.9F + static_cast<float>(lane % 5) * 0.02F;
    key_scale[lane] = 0.85F + static_cast<float>(lane % 6) * 0.02F;
  }
  std::vector<__nv_bfloat16> committed_logical(cache_values);
  for (std::size_t index = 0; index < cache_values; ++index) {
    committed_logical[index] = __float2bfloat16_rn(
        static_cast<float>(static_cast<int>(index % 31) - 15) * 0.015625F);
  }
  std::vector<__nv_bfloat16> committed_physical(cache_values);
  qw38::cuda::attention_kv_copy_logical_to_physical(
      committed_logical.data(), committed_physical.data(), kConfig.kv_heads,
      kConfig.capacity, kConfig.head_width);

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
  QW38_ALLOC(device_normalized_query, qw38::cuda::attention_query_values(kConfig));
  QW38_ALLOC(device_normalized_key, qw38::cuda::attention_kv_row_values(kConfig));
  QW38_ALLOC(device_scores, score_values);
  QW38_ALLOC(device_output, query_values);
  QW38_ALLOC(device_tiled, query_values);
  QW38_ALLOC(device_partial, partial_values);
  QW38_ALLOC(device_meta, meta_values);
  QW38_ALLOC(device_committed_key, cache_values);
  QW38_ALLOC(device_committed_value, cache_values);
  QW38_ALLOC(device_candidate_key, row_values);
  QW38_ALLOC(device_candidate_value, row_values);
#undef QW38_ALLOC
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ab cudaMalloc", error);
  }
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
  QW38_COPY(device_committed_key, committed_physical);
  QW38_COPY(device_committed_value, committed_physical);
#undef QW38_COPY
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ab H2D", error);
  }

  const qw38::cuda::AttentionCache committed{device_committed_key,
                                            device_committed_value};
  const qw38::cuda::AttentionCache candidate{device_candidate_key,
                                            device_candidate_value};
  error = qw38::cuda::launch_attention_prepare_chunk_tiled(
      kConfig, 0, kTokenCount, device_query, device_key, device_value,
      device_query_scale, device_key_scale, device_gate, committed, candidate,
      device_normalized_query, device_normalized_key, device_scores,
      device_tiled, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ab tiled", error);
  }
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
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ab tiled candidate", error);
  }
  std::vector<float> tiled(query_values);
  error = cudaMemcpy(tiled.data(), device_tiled, query_values * sizeof(float),
                      cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ab tiled D2H", error);
  }

  struct CandidateResult {
    const char* id = "global";
    int occupancy = 0;
    Envelope vs_tiled;
    bool byte_equal = false;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool launch_ok = false;
    bool eligible = false;
    std::vector<float> warmup_ms;
    std::vector<float> samples;
    float mean_ms = 0.0F;
    std::vector<float> output;
  };
  CandidateResult results[kCandidateCount];
  std::vector<float> global_output;

  for (int c = 0; c < kCandidateCount; ++c) {
    const char* id = kCandidates[c];
    results[c].id = id;
    results[c].occupancy = occupancy_for(id);
    error = cudaMemcpy(device_scores, sentinel.data(),
                        score_values * sizeof(float), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_committed_key, committed_physical.data(),
                         cache_values * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_committed_value, committed_physical.data(),
                         cache_values * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("ab restore", error);
    }
    error = qw38::cuda::launch_attention_prepare_chunk_stream_k_vkq(
        kConfig, 0, kTokenCount, device_query, device_key, device_value,
        device_query_scale, device_key_scale, device_gate, committed,
        candidate, device_normalized_query, device_normalized_key,
        device_scores, device_output, device_partial, device_meta, id,
        nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    results[c].launch_ok = error == cudaSuccess;
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("ab candidate launch", error);
    }
    results[c].output.assign(query_values, 0.0F);
    std::vector<float> scores(score_values);
    std::vector<__nv_bfloat16> actual_candidate_key(row_values);
    std::vector<__nv_bfloat16> actual_candidate_value(row_values);
    std::vector<__nv_bfloat16> committed_key_after(cache_values);
    std::vector<__nv_bfloat16> committed_value_after(cache_values);
    error = cudaMemcpy(results[c].output.data(), device_output,
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
    if (error == cudaSuccess) {
      error = cudaMemcpy(committed_key_after.data(), device_committed_key,
                         cache_values * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(committed_value_after.data(), device_committed_value,
                         cache_values * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("ab candidate D2H", error);
    }
    results[c].vs_tiled = compare(results[c].output, tiled);
    results[c].scratch_unchanged =
        std::memcmp(scores.data(), sentinel.data(),
                    score_values * sizeof(float)) == 0;
    results[c].committed_unchanged =
        std::memcmp(committed_key_after.data(), committed_physical.data(),
                    cache_values * sizeof(__nv_bfloat16)) == 0 &&
        std::memcmp(committed_value_after.data(), committed_physical.data(),
                    cache_values * sizeof(__nv_bfloat16)) == 0;
    results[c].candidate_exact =
        std::memcmp(actual_candidate_key.data(), tiled_candidate_key.data(),
                    row_values * sizeof(__nv_bfloat16)) == 0 &&
        std::memcmp(actual_candidate_value.data(), tiled_candidate_value.data(),
                    row_values * sizeof(__nv_bfloat16)) == 0;
    if (c == 0) {
      global_output = results[c].output;
      results[c].byte_equal = true;
    } else {
      results[c].byte_equal =
          std::memcmp(results[c].output.data(), global_output.data(),
                      query_values * sizeof(float)) == 0;
    }
    results[c].eligible =
        results[c].launch_ok && results[c].occupancy >= 1 &&
        results[c].vs_tiled.nonfinite == 0 &&
        results[c].vs_tiled.max_abs <= 5.0e-5F &&
        results[c].vs_tiled.rms <= 5.0e-6F && results[c].scratch_unchanged &&
        results[c].committed_unchanged && results[c].candidate_exact &&
        (c == 0 || results[c].byte_equal);
    std::fprintf(raw,
                 "vkq_ab id=%s occupancy=%d launch_ok=%s byte_equal=%s "
                 "max_abs=%.9g rms=%.9g nonfinite=%zu scratch=%s "
                 "committed=%s candidate=%s eligible=%s\n",
                 id, results[c].occupancy, json_bool(results[c].launch_ok),
                 json_bool(results[c].byte_equal),
                 static_cast<double>(results[c].vs_tiled.max_abs),
                 static_cast<double>(results[c].vs_tiled.rms),
                 results[c].vs_tiled.nonfinite,
                 json_bool(results[c].scratch_unchanged),
                 json_bool(results[c].committed_unchanged),
                 json_bool(results[c].candidate_exact),
                 json_bool(results[c].eligible));
  }
  if (!results[0].eligible) {
    std::fclose(raw);
    std::fprintf(stderr, "global stream-K candidate is not eligible\n");
    return 1;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ab events", error);
  }
  for (int round = 0; round < kWarmups + kMeasured; ++round) {
    for (int c = 0; c < kCandidateCount; ++c) {
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_attention_prepare_chunk_stream_k_vkq(
            kConfig, 0, kTokenCount, device_query, device_key, device_value,
            device_query_scale, device_key_scale, device_gate, committed,
            candidate, device_normalized_query, device_normalized_key,
            device_scores, device_output, device_partial, device_meta,
            kCandidates[c], nullptr);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float ms = 0.0F;
      if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
      if (error != cudaSuccess) {
        std::fclose(raw);
        return fail_cuda("ab timed launch", error);
      }
      if (round < kWarmups) {
        results[c].warmup_ms.push_back(ms);
        std::fprintf(raw, "vkq_ab id=%s warmup=%d ms=%.9g occupancy=%d\n",
                     kCandidates[c], round, static_cast<double>(ms),
                     results[c].occupancy);
      } else {
        results[c].samples.push_back(ms);
        std::fprintf(raw, "vkq_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
                     kCandidates[c], round - kWarmups, static_cast<double>(ms),
                     results[c].occupancy);
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  for (int c = 0; c < kCandidateCount; ++c) {
    double sum = 0.0;
    for (float sample : results[c].samples) sum += sample;
    results[c].mean_ms =
        static_cast<float>(sum / static_cast<double>(results[c].samples.size()));
    std::fprintf(raw,
                 "vkq_ab_mean id=%s mean_ms=%.9g occupancy=%d eligible=%s\n",
                 results[c].id, static_cast<double>(results[c].mean_ms),
                 results[c].occupancy, json_bool(results[c].eligible));
  }
  const float global_mean = results[0].mean_ms;
  const bool registers_win =
      results[1].eligible && results[1].mean_ms < global_mean;
  const char* winner = registers_win ? "registers" : "global";
  std::fprintf(raw,
               "vkq_ab_winner id=%s win=%s global_ms=%.9g registers_ms=%.9g "
               "include_combine=true\n",
               winner, json_bool(registers_win),
               static_cast<double>(global_mean),
               static_cast<double>(results[1].mean_ms));

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-033\",\"status\":\"passed\",",
              kPrefix);
  std::printf("\"device\":\"%s\",\"compute_capability\":\"%d.%d\",", prop.name,
              prop.major, prop.minor);
  std::printf("\"measurement_utc\":\"%s\",", utc);
  std::printf("\"include_combine\":true,\"winner\":\"%s\",\"win\":%s,", winner,
              json_bool(registers_win));
  std::printf("\"candidates\":{");
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::printf(",");
    const CandidateResult& result = results[c];
    std::printf("\"%s\":{\"id\":\"%s\",\"mean_ms\":%.9g,", result.id, result.id,
               static_cast<double>(result.mean_ms));
    print_array(stdout, "samples", result.samples);
    std::printf(",");
    print_array(stdout, "warmup_ms", result.warmup_ms);
    std::printf(
        ",\"occupancy\":%d,\"launch_ok\":%s,\"eligible\":%s,\"byte_equal\":%s,"
        "\"scratch_unchanged\":%s,\"committed_unchanged\":%s,"
        "\"candidate_exact\":%s,\"include_combine\":true,"
        "\"vs_tiled\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu}}",
        result.occupancy, json_bool(result.launch_ok),
        json_bool(result.eligible), json_bool(result.byte_equal),
        json_bool(result.scratch_unchanged),
        json_bool(result.committed_unchanged), json_bool(result.candidate_exact),
        static_cast<double>(result.vs_tiled.max_abs),
        static_cast<double>(result.vs_tiled.rms), result.vs_tiled.nonfinite);
  }
  std::printf("},\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\"}\n");
  std::printf("status=passed\n");
  std::fclose(raw);

  cudaFree(device_candidate_value);
  cudaFree(device_candidate_key);
  cudaFree(device_committed_value);
  cudaFree(device_committed_key);
  cudaFree(device_meta);
  cudaFree(device_partial);
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
