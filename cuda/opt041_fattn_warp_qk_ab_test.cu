#include "attention_decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_FATTN_WARP_QK_AB_RESULT=";
constexpr const char* kCandidates[] = {"cparts", "warp_microtile"};
constexpr int kCandidateCount = 2;
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr std::size_t kTokenCount = 4096;
constexpr std::size_t kCorrectnessTokens[] = {16, 17, 31, 32, 33, 63, 64, 65,
                                              512, 2048, 4096};
constexpr std::size_t kCorrectnessStarts[] = {0, 1, 31, 128, 2048};
constexpr int kCorrectnessTokenCount = 11;
constexpr int kCorrectnessStartCount = 5;
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

int occupancy_for(const char* qk_path) {
  if (std::strcmp(qk_path, "warp_microtile") == 0)
    return qw38::cuda::fattn_warp_qk_occupancy();
  return qw38::cuda::fattn_pv_mma_occupancy();
}

std::size_t query_count(std::size_t token_count) {
  return token_count * qw38::cuda::attention_query_values(kConfig);
}

std::size_t row_count(std::size_t token_count) {
  return token_count * qw38::cuda::attention_kv_row_values(kConfig);
}

}  // namespace

int main(int argc, char** argv) {
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt041-fattn-warp-qk/warp-qk-ab-raw.txt";
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  const std::size_t query_values = query_count(kTokenCount);
  const std::size_t row_values = row_count(kTokenCount);
  const std::size_t cache_values = qw38::cuda::attention_cache_values(kConfig);
  const std::size_t max_score_values =
      qw38::cuda::attention_chunk_score_values(kConfig, 2048, kTokenCount);
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
  std::vector<float> sentinel(max_score_values, 0.125F);
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
  QW38_ALLOC(device_scores, max_score_values);
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

  struct LaunchMetrics {
    bool launch_ok = false;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool outputs_byte_equal = false;
    Envelope vs_tiled;
    Envelope vs_cparts;
    std::vector<float> output;
  };

  auto restore = [&](std::size_t token_count, std::size_t start_position)
      -> cudaError_t {
    const std::size_t scores = qw38::cuda::attention_chunk_score_values(
        kConfig, start_position, token_count);
    const std::size_t qn = query_count(token_count);
    const std::size_t rn = row_count(token_count);
    cudaError_t local = cudaMemcpy(device_query, query.data(),
                                   qn * sizeof(float), cudaMemcpyHostToDevice);
    if (local == cudaSuccess)
      local = cudaMemcpy(device_key, key.data(), rn * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (local == cudaSuccess)
      local = cudaMemcpy(device_value, value.data(), rn * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (local == cudaSuccess)
      local = cudaMemcpy(device_gate, gate.data(), qn * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (local == cudaSuccess)
      local = cudaMemcpy(device_scores, sentinel.data(), scores * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (local == cudaSuccess)
      local = cudaMemcpy(device_committed_key, committed_physical.data(),
                         cache_values * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    if (local == cudaSuccess)
      local = cudaMemcpy(device_committed_value, committed_physical.data(),
                         cache_values * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    return local;
  };

  auto launch_one = [&](std::size_t start_position, std::size_t token_count,
                        const char* qk_path, LaunchMetrics* metrics,
                        const std::vector<float>* cparts_output,
                        const std::vector<__nv_bfloat16>* staging_key,
                        const std::vector<__nv_bfloat16>* staging_value)
      -> cudaError_t {
    cudaError_t local = restore(token_count, start_position);
    if (local == cudaSuccess) {
      local = qw38::cuda::launch_attention_prepare_chunk_stream_k_qk(
          kConfig, start_position, token_count, device_query, device_key,
          device_value, device_query_scale, device_key_scale, device_gate,
          committed, candidate, device_normalized_query, device_normalized_key,
          device_scores, device_output, device_partial, device_meta, qk_path,
          nullptr);
    }
    if (local == cudaSuccess) local = cudaDeviceSynchronize();
    metrics->launch_ok = local == cudaSuccess;
    if (local != cudaSuccess) return local;
    const std::size_t qn = query_count(token_count);
    const std::size_t rn = row_count(token_count);
    const std::size_t scores = qw38::cuda::attention_chunk_score_values(
        kConfig, start_position, token_count);
    metrics->output.assign(qn, 0.0F);
    std::vector<float> score_host(scores);
    std::vector<__nv_bfloat16> actual_key(rn);
    std::vector<__nv_bfloat16> actual_value(rn);
    std::vector<__nv_bfloat16> committed_key_after(cache_values);
    std::vector<__nv_bfloat16> committed_value_after(cache_values);
    local = cudaMemcpy(metrics->output.data(), device_output,
                       qn * sizeof(float), cudaMemcpyDeviceToHost);
    if (local == cudaSuccess)
      local = cudaMemcpy(score_host.data(), device_scores,
                         scores * sizeof(float), cudaMemcpyDeviceToHost);
    if (local == cudaSuccess)
      local = cudaMemcpy(actual_key.data(), device_candidate_key,
                         rn * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    if (local == cudaSuccess)
      local = cudaMemcpy(actual_value.data(), device_candidate_value,
                         rn * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    if (local == cudaSuccess)
      local = cudaMemcpy(committed_key_after.data(), device_committed_key,
                         cache_values * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    if (local == cudaSuccess)
      local = cudaMemcpy(committed_value_after.data(), device_committed_value,
                         cache_values * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    if (local != cudaSuccess) return local;
    metrics->scratch_unchanged =
        std::memcmp(score_host.data(), sentinel.data(),
                    scores * sizeof(float)) == 0;
    metrics->committed_unchanged =
        std::memcmp(committed_key_after.data(), committed_physical.data(),
                    cache_values * sizeof(__nv_bfloat16)) == 0 &&
        std::memcmp(committed_value_after.data(), committed_physical.data(),
                    cache_values * sizeof(__nv_bfloat16)) == 0;
    if (staging_key != nullptr && staging_value != nullptr) {
      metrics->candidate_exact =
          std::memcmp(actual_key.data(), staging_key->data(),
                      rn * sizeof(__nv_bfloat16)) == 0 &&
          std::memcmp(actual_value.data(), staging_value->data(),
                      rn * sizeof(__nv_bfloat16)) == 0;
    } else {
      metrics->candidate_exact = true;
    }
    if (start_position == 0 && token_count == kTokenCount) {
      metrics->vs_tiled = compare(metrics->output, tiled);
    }
    if (cparts_output != nullptr) {
      metrics->vs_cparts = compare(metrics->output, *cparts_output);
      metrics->outputs_byte_equal =
          metrics->output.size() == cparts_output->size() &&
          std::memcmp(metrics->output.data(), cparts_output->data(),
                      qn * sizeof(float)) == 0;
    } else {
      metrics->outputs_byte_equal = true;
    }
    return cudaSuccess;
  };

  struct CandidateResult {
    const char* id = "cparts";
    int occupancy = 0;
    Envelope vs_tiled;
    Envelope vs_cparts;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool outputs_byte_equal = false;
    bool launch_ok = false;
    bool eligible = false;
    std::vector<float> warmup_ms;
    std::vector<float> samples;
    float mean_ms = 0.0F;
    std::vector<float> output;
  };
  CandidateResult results[kCandidateCount];
  std::vector<float> cparts_output;
  std::vector<__nv_bfloat16> staging_key = tiled_candidate_key;
  std::vector<__nv_bfloat16> staging_value = tiled_candidate_value;

  for (int c = 0; c < kCandidateCount; ++c) {
    const char* id = kCandidates[c];
    results[c].id = id;
    results[c].occupancy = occupancy_for(id);
    LaunchMetrics metrics;
    const std::vector<float>* baseline =
        c == 0 ? nullptr : &cparts_output;
    error = launch_one(0, kTokenCount, id, &metrics, baseline, &staging_key,
                       &staging_value);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("ab candidate launch", error);
    }
    results[c].launch_ok = metrics.launch_ok;
    results[c].output = metrics.output;
    results[c].vs_tiled = metrics.vs_tiled;
    results[c].vs_cparts = metrics.vs_cparts;
    results[c].scratch_unchanged = metrics.scratch_unchanged;
    results[c].committed_unchanged = metrics.committed_unchanged;
    results[c].candidate_exact = metrics.candidate_exact;
    results[c].outputs_byte_equal = metrics.outputs_byte_equal;
    if (c == 0) cparts_output = metrics.output;
    const bool envelope_ok = results[c].vs_tiled.nonfinite == 0 &&
                             results[c].vs_tiled.max_abs <= 5.0e-5F &&
                             results[c].vs_tiled.rms <= 5.0e-6F;
    results[c].eligible =
        results[c].launch_ok && results[c].occupancy >= 1 && envelope_ok &&
        results[c].scratch_unchanged && results[c].committed_unchanged &&
        results[c].candidate_exact && results[c].outputs_byte_equal;
    std::fprintf(raw,
                 "warp_qk_ab id=%s occupancy=%d launch_ok=%s "
                 "max_abs=%.9g rms=%.9g nonfinite=%zu vs_cparts_max_abs=%.9g "
                 "vs_cparts_rms=%.9g scratch=%s committed=%s candidate=%s "
                 "byte_equal=%s eligible=%s\n",
                 id, results[c].occupancy, json_bool(results[c].launch_ok),
                 static_cast<double>(results[c].vs_tiled.max_abs),
                 static_cast<double>(results[c].vs_tiled.rms),
                 results[c].vs_tiled.nonfinite,
                 static_cast<double>(results[c].vs_cparts.max_abs),
                 static_cast<double>(results[c].vs_cparts.rms),
                 json_bool(results[c].scratch_unchanged),
                 json_bool(results[c].committed_unchanged),
                 json_bool(results[c].candidate_exact),
                 json_bool(results[c].outputs_byte_equal),
                 json_bool(results[c].eligible));
  }
  if (!results[0].eligible) {
    std::fclose(raw);
    std::fprintf(stderr, "cparts stream-K candidate is not eligible\n");
    return 1;
  }

  struct CaseMetrics {
    std::size_t start_position = 0;
    std::size_t token_count = 0;
    bool launch_ok = false;
    bool outputs_byte_equal = false;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool eligible = false;
    Envelope vs_cparts;
  };
  std::vector<CaseMetrics> cases(static_cast<std::size_t>(kCorrectnessTokenCount *
                                                          kCorrectnessStartCount));
  bool all_eligible = true;
  for (int start_index = 0; start_index < kCorrectnessStartCount; ++start_index) {
    for (int token_index = 0; token_index < kCorrectnessTokenCount;
         ++token_index) {
      const std::size_t start = kCorrectnessStarts[start_index];
      const std::size_t tokens = kCorrectnessTokens[token_index];
      const std::size_t slot = static_cast<std::size_t>(
          start_index * kCorrectnessTokenCount + token_index);
      CaseMetrics& item = cases[slot];
      item.start_position = start;
      item.token_count = tokens;
      if (start + tokens > kConfig.capacity) {
        item.eligible = false;
        all_eligible = false;
        continue;
      }
      LaunchMetrics cparts;
      LaunchMetrics warp;
      error = launch_one(start, tokens, "cparts", &cparts, nullptr, nullptr,
                         nullptr);
      if (error == cudaSuccess) {
        std::vector<__nv_bfloat16> case_key(row_count(tokens));
        std::vector<__nv_bfloat16> case_value(row_count(tokens));
        error = cudaMemcpy(case_key.data(), device_candidate_key,
                           row_count(tokens) * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
        if (error == cudaSuccess)
          error = cudaMemcpy(case_value.data(), device_candidate_value,
                             row_count(tokens) * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost);
        if (error == cudaSuccess)
          error = launch_one(start, tokens, "warp_microtile", &warp,
                             &cparts.output, &case_key, &case_value);
      }
      item.launch_ok = error == cudaSuccess && cparts.launch_ok && warp.launch_ok;
      item.outputs_byte_equal = warp.outputs_byte_equal;
      item.scratch_unchanged =
          cparts.scratch_unchanged && warp.scratch_unchanged;
      item.committed_unchanged =
          cparts.committed_unchanged && warp.committed_unchanged;
      item.candidate_exact = warp.candidate_exact;
      item.vs_cparts = warp.vs_cparts;
      item.eligible = item.launch_ok && item.outputs_byte_equal &&
                      item.scratch_unchanged && item.committed_unchanged &&
                      item.candidate_exact && item.vs_cparts.nonfinite == 0;
      all_eligible = all_eligible && item.eligible;
      std::fprintf(raw,
                   "correctness start=%zu tokens=%zu launch_ok=%s byte_equal=%s "
                   "eligible=%s vs_cparts_max_abs=%.9g\n",
                   start, tokens, json_bool(item.launch_ok),
                   json_bool(item.outputs_byte_equal), json_bool(item.eligible),
                   static_cast<double>(item.vs_cparts.max_abs));
    }
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
      error = restore(kTokenCount, 0);
      if (error == cudaSuccess) error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_attention_prepare_chunk_stream_k_qk(
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
        std::fprintf(raw, "warp_qk_ab id=%s warmup=%d ms=%.9g occupancy=%d\n",
                     kCandidates[c], round, static_cast<double>(ms),
                     results[c].occupancy);
      } else {
        results[c].samples.push_back(ms);
        std::fprintf(raw, "warp_qk_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
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
                 "warp_qk_ab_mean id=%s mean_ms=%.9g occupancy=%d eligible=%s\n",
                 results[c].id, static_cast<double>(results[c].mean_ms),
                 results[c].occupancy, json_bool(results[c].eligible));
  }
  const float cparts_mean = results[0].mean_ms;
  const bool warp_win = all_eligible && results[1].eligible &&
                        results[1].mean_ms < cparts_mean;
  const char* winner = warp_win ? "warp_microtile" : "cparts";
  std::fprintf(raw,
               "warp_qk_ab_winner id=%s win=%s cparts_ms=%.9g "
               "warp_microtile_ms=%.9g include_combine=true all_eligible=%s\n",
               winner, json_bool(warp_win), static_cast<double>(cparts_mean),
               static_cast<double>(results[1].mean_ms), json_bool(all_eligible));

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-041\",\"status\":\"passed\","
      "\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"measurement_utc\":\"%s\",\"include_combine\":true,"
      "\"selected_vkq_accum\":\"registers\",\"selected_pv_path\":\"mma\","
      "\"ab_p4096\":{\"token_count\":4096,\"winner\":\"%s\",\"win\":%s,"
      "\"include_combine\":true,\"candidates\":{",
      kPrefix, prop.name, prop.major, prop.minor, utc, winner,
      json_bool(warp_win));
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::printf(",");
    const CandidateResult& result = results[c];
    std::printf("\"%s\":{\"id\":\"%s\",\"mean_ms\":%.9g,", result.id, result.id,
               static_cast<double>(result.mean_ms));
    print_array(stdout, "samples", result.samples);
    std::printf(",");
    print_array(stdout, "warmup_ms", result.warmup_ms);
    std::printf(
        ",\"occupancy\":%d,\"launch_ok\":%s,\"eligible\":%s,"
        "\"scratch_unchanged\":%s,\"committed_unchanged\":%s,"
        "\"candidate_exact\":%s,\"outputs_byte_equal\":%s,"
        "\"include_combine\":true,"
        "\"vs_tiled\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu},"
        "\"vs_cparts\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu}}",
        result.occupancy, json_bool(result.launch_ok),
        json_bool(result.eligible), json_bool(result.scratch_unchanged),
        json_bool(result.committed_unchanged), json_bool(result.candidate_exact),
        json_bool(result.outputs_byte_equal),
        static_cast<double>(result.vs_tiled.max_abs),
        static_cast<double>(result.vs_tiled.rms), result.vs_tiled.nonfinite,
        static_cast<double>(result.vs_cparts.max_abs),
        static_cast<double>(result.vs_cparts.rms), result.vs_cparts.nonfinite);
  }
  std::printf(
      "}},\"correctness\":{\"token_counts\":[16,17,31,32,33,63,64,65,512,2048,"
      "4096],\"starts\":[0,1,31,128,2048],\"all_eligible\":%s,\"cases\":{",
      json_bool(all_eligible));
  bool first = true;
  for (const CaseMetrics& item : cases) {
    if (!first) std::printf(",");
    first = false;
    std::printf(
        "\"%zu:%zu\":{\"start_position\":%zu,\"token_count\":%zu,"
        "\"launch_ok\":%s,\"outputs_byte_equal\":%s,"
        "\"scratch_unchanged\":%s,\"committed_unchanged\":%s,"
        "\"candidate_exact\":%s,\"eligible\":%s,"
        "\"vs_cparts\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu}}",
        item.start_position, item.token_count, item.start_position,
        item.token_count, json_bool(item.launch_ok),
        json_bool(item.outputs_byte_equal), json_bool(item.scratch_unchanged),
        json_bool(item.committed_unchanged), json_bool(item.candidate_exact),
        json_bool(item.eligible), static_cast<double>(item.vs_cparts.max_abs),
        static_cast<double>(item.vs_cparts.rms), item.vs_cparts.nonfinite);
  }
  std::printf("}},\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\"}\n");
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
