#include "attention_decode.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT051_ATTENTION_PIPELINE_AB_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr const char* kCandidates[] = {
    "off",        "f16",       "dual_reg", "f16_reg",
    "dual_async", "f16_async", "nbatch64", "gqa6"};
constexpr int kCandidateCount = 8;
constexpr qw38::cuda::AttentionConfig kConfig{24, 4, 256, 64, 131072};
constexpr std::size_t kCorrectnessTokens[] = {1, 16, 17, 31, 32, 33, 63, 64, 65,
                                              512, 2048, 4096};
constexpr std::size_t kCorrectnessStarts[] = {0, 1, 31, 128, 2048};
constexpr int kCorrectnessTokenCount = 12;
constexpr int kCorrectnessStartCount = 5;
constexpr float kOpt044Abs = 3.0e-4F;
constexpr float kOpt044Rms = 2.0e-4F;

struct LayerSpec {
  const char* id;
  std::size_t q_off;
  std::size_t k_off;
};

constexpr LayerSpec kLayers[] = {
    {"layer3", 2688419808ULL, 2595746784ULL},
    {"layer31", 10215498080ULL, 10122825056ULL},
    {"layer63", 18817873248ULL, 18725200224ULL},
};
constexpr int kLayerCount = 3;

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
  float one_minus_cosine = 0.0F;
  std::size_t nonfinite = 0;
};

Envelope compare(const std::vector<float>& actual,
                 const std::vector<float>& expected) {
  Envelope metrics;
  double squared = 0.0;
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  const std::size_t n = actual.size();
  for (std::size_t index = 0; index < n; ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
      ++metrics.nonfinite;
    const float error = std::fabs(actual[index] - expected[index]);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
    dot += static_cast<double>(actual[index]) * expected[index];
    na += static_cast<double>(actual[index]) * actual[index];
    nb += static_cast<double>(expected[index]) * expected[index];
  }
  metrics.rms = n == 0 ? 0.0F
                       : static_cast<float>(std::sqrt(squared / static_cast<double>(n)));
  const double denom = std::sqrt(na) * std::sqrt(nb);
  const double cosine = denom == 0.0 ? 1.0 : dot / denom;
  metrics.one_minus_cosine = static_cast<float>(1.0 - cosine);
  return metrics;
}

std::size_t query_count(std::size_t token_count) {
  return token_count * qw38::cuda::attention_query_values(kConfig);
}

std::size_t row_count(std::size_t token_count) {
  return token_count * qw38::cuda::attention_kv_row_values(kConfig);
}

int occupancy_for(const char* path) {
  const int prep = qw38::cuda::attention_prepare_query_occupancy();
  const int attn = qw38::cuda::fattn_pipeline_occupancy_path(path);
  return prep < attn ? prep : attn;
}

bool like_arithmetic(const char* path) {
  return std::strcmp(path, "off") == 0;
}

int index_of(const char* path) {
  for (int index = 0; index < kCandidateCount; ++index) {
    if (std::strcmp(kCandidates[index], path) == 0) return index;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set to smoke, "
                         "correctness, or acceptance\n");
    return 1;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  const int warmups = qw38::cuda::test_warmups();
  const int measured = qw38::cuda::test_samples();
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt051-attention-pipeline/"
                 "attention-pipeline-ab-raw.txt";
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  const std::size_t timed_tokens = tier == qw38::cuda::TestTier::kSmoke ? 16 : 4096;
  const std::size_t query_values = query_count(timed_tokens);
  const std::size_t row_values = row_count(timed_tokens);
  const std::size_t cache_values = qw38::cuda::attention_cache_values(kConfig);
  const std::size_t max_score_values =
      qw38::cuda::attention_chunk_score_values(kConfig, 2048, timed_tokens);
  const std::size_t partial_values =
      qw38::cuda::fattn_stream_k_partial_values(kConfig, timed_tokens);
  const std::size_t meta_values =
      qw38::cuda::fattn_stream_k_meta_values(kConfig, timed_tokens);
  const std::size_t prepared_halfs =
      qw38::cuda::attention_prepared_query_bytes(kConfig, timed_tokens) /
      sizeof(__half);

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
  float* device_partial = nullptr;
  float* device_meta = nullptr;
  __half* device_prepared = nullptr;
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
  QW38_ALLOC(device_partial, partial_values);
  QW38_ALLOC(device_meta, meta_values);
  QW38_ALLOC(device_prepared, prepared_halfs);
  QW38_ALLOC(device_committed_key, cache_values);
  QW38_ALLOC(device_committed_value, cache_values);
  QW38_ALLOC(device_candidate_key, row_values);
  QW38_ALLOC(device_candidate_value, row_values);
#undef QW38_ALLOC
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("alloc", error);
  }

  const qw38::cuda::AttentionCache committed{device_committed_key,
                                             device_committed_value};
  const qw38::cuda::AttentionCache candidate{device_candidate_key,
                                             device_candidate_value};

  auto restore = [&](std::size_t token_count, std::size_t start_position)
      -> cudaError_t {
    (void)start_position;
    const std::size_t qn = query_count(token_count);
    const std::size_t rn = row_count(token_count);
    const std::size_t scores = qw38::cuda::attention_chunk_score_values(
        kConfig, start_position, token_count);
    cudaError_t local =
        cudaMemcpy(device_query, query.data(), qn * sizeof(float),
                   cudaMemcpyHostToDevice);
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
      local = cudaMemcpy(device_query_scale, query_scale.data(),
                         kConfig.head_width * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (local == cudaSuccess)
      local = cudaMemcpy(device_key_scale, key_scale.data(),
                         kConfig.head_width * sizeof(float),
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

  struct LaunchMetrics {
    Envelope vs_off;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool outputs_byte_equal = false;
    bool graph_eager_equal = false;
    bool launch_ok = false;
    std::vector<float> output;
  };

  auto launch_one = [&](std::size_t start_position, std::size_t token_count,
                        const char* path, LaunchMetrics* metrics,
                        const std::vector<float>* baseline,
                        const std::vector<__nv_bfloat16>* staging_key,
                        const std::vector<__nv_bfloat16>* staging_value,
                        bool capture_graph) -> cudaError_t {
    qw38::cuda::QueryPreparePathScope prep("hoisted");
    qw38::cuda::AttentionPipelinePathScope pipe(path);
    cudaError_t local = restore(token_count, start_position);
    auto run = [&]() -> cudaError_t {
      return qw38::cuda::launch_attention_prepare_chunk_stream_k(
          kConfig, start_position, token_count, device_query, device_key,
          device_value, device_query_scale, device_key_scale, device_gate,
          committed, candidate, device_normalized_query, device_normalized_key,
          device_scores, device_output, device_partial, device_meta, nullptr,
          device_prepared);
    };
    if (local == cudaSuccess) {
      if (capture_graph) {
        cudaStream_t stream = nullptr;
        local = cudaStreamCreate(&stream);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        if (local == cudaSuccess)
          local = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (local == cudaSuccess)
          local = qw38::cuda::launch_attention_prepare_chunk_stream_k(
              kConfig, start_position, token_count, device_query, device_key,
              device_value, device_query_scale, device_key_scale, device_gate,
              committed, candidate, device_normalized_query,
              device_normalized_key, device_scores, device_output,
              device_partial, device_meta, stream, device_prepared);
        if (local == cudaSuccess) local = cudaStreamEndCapture(stream, &graph);
        if (local == cudaSuccess)
          local = cudaGraphInstantiate(&exec, graph, 0);
        std::vector<float> eager;
        if (local == cudaSuccess) {
          local = restore(token_count, start_position);
          if (local == cudaSuccess) local = run();
          if (local == cudaSuccess) local = cudaDeviceSynchronize();
          eager.assign(query_count(token_count), 0.0F);
          if (local == cudaSuccess)
            local = cudaMemcpy(eager.data(), device_output,
                               query_count(token_count) * sizeof(float),
                               cudaMemcpyDeviceToHost);
        }
        if (local == cudaSuccess) local = restore(token_count, start_position);
        if (local == cudaSuccess) local = cudaGraphLaunch(exec, nullptr);
        if (local == cudaSuccess) local = cudaDeviceSynchronize();
        std::vector<float> replay(query_count(token_count));
        if (local == cudaSuccess)
          local = cudaMemcpy(replay.data(), device_output,
                             query_count(token_count) * sizeof(float),
                             cudaMemcpyDeviceToHost);
        metrics->graph_eager_equal =
            local == cudaSuccess && eager.size() == replay.size() &&
            std::memcmp(eager.data(), replay.data(),
                        eager.size() * sizeof(float)) == 0;
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (stream != nullptr) cudaStreamDestroy(stream);
        metrics->output = std::move(replay);
      } else {
        local = run();
        if (local == cudaSuccess) local = cudaDeviceSynchronize();
        metrics->graph_eager_equal = true;
      }
    }
    metrics->launch_ok = local == cudaSuccess;
    if (local != cudaSuccess) return local;
    const std::size_t qn = query_count(token_count);
    const std::size_t rn = row_count(token_count);
    const std::size_t scores = qw38::cuda::attention_chunk_score_values(
        kConfig, start_position, token_count);
    if (metrics->output.empty()) metrics->output.assign(qn, 0.0F);
    std::vector<float> score_host(scores);
    std::vector<__nv_bfloat16> actual_key(rn);
    std::vector<__nv_bfloat16> actual_value(rn);
    std::vector<__nv_bfloat16> committed_key_after(cache_values);
    std::vector<__nv_bfloat16> committed_value_after(cache_values);
    if (!capture_graph) {
      local = cudaMemcpy(metrics->output.data(), device_output, qn * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
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
    if (baseline != nullptr) {
      metrics->vs_off = compare(metrics->output, *baseline);
      metrics->outputs_byte_equal =
          metrics->output.size() == baseline->size() &&
          std::memcmp(metrics->output.data(), baseline->data(),
                      qn * sizeof(float)) == 0;
    } else {
      metrics->outputs_byte_equal = true;
    }
    return cudaSuccess;
  };

  struct CandidateResult {
    const char* id = "off";
    int occupancy = 0;
    Envelope vs_off;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool outputs_byte_equal = false;
    bool graph_eager_equal = false;
    bool launch_ok = false;
    bool eligible = false;
    std::vector<float> warmup_ms;
    std::vector<float> samples;
    float mean_ms = 0.0F;
    std::vector<float> output;
  };
  CandidateResult results[kCandidateCount];
  std::vector<float> off_output;
  std::vector<__nv_bfloat16> staging_key;
  std::vector<__nv_bfloat16> staging_value;

  for (int c = 0; c < kCandidateCount; ++c) {
    const char* id = kCandidates[c];
    results[c].id = id;
    results[c].occupancy = occupancy_for(id);
    if (results[c].occupancy < 1 && std::strcmp(id, "off") != 0) {
      results[c].launch_ok = false;
      results[c].eligible = false;
      std::fprintf(raw,
                   "pipeline_ab id=%s occupancy=%d launch_ok=false eligible=false "
                   "skip=occupancy\n",
                   id, results[c].occupancy);
      continue;
    }
    LaunchMetrics metrics;
    const std::vector<float>* baseline = c == 0 ? nullptr : &off_output;
    const std::vector<__nv_bfloat16>* sk = c == 0 ? nullptr : &staging_key;
    const std::vector<__nv_bfloat16>* sv = c == 0 ? nullptr : &staging_value;
    error = launch_one(0, timed_tokens, id, &metrics, baseline, sk, sv,
                       c != 0 && timed_tokens >= 16);
    if (error != cudaSuccess) {
      std::fclose(raw);
      std::fprintf(stderr, "ab candidate launch id=%s occupancy=%d: %s\n", id,
                   results[c].occupancy, cudaGetErrorString(error));
      return fail_cuda("ab candidate launch", error);
    }
    results[c].launch_ok = metrics.launch_ok;
    results[c].output = metrics.output;
    results[c].vs_off = metrics.vs_off;
    results[c].scratch_unchanged = metrics.scratch_unchanged;
    results[c].committed_unchanged = metrics.committed_unchanged;
    results[c].candidate_exact = metrics.candidate_exact;
    results[c].outputs_byte_equal = metrics.outputs_byte_equal;
    results[c].graph_eager_equal = metrics.graph_eager_equal;
    if (c == 0) {
      off_output = metrics.output;
      staging_key.resize(row_count(timed_tokens));
      staging_value.resize(row_count(timed_tokens));
      error = cudaMemcpy(staging_key.data(), device_candidate_key,
                         row_count(timed_tokens) * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
      if (error == cudaSuccess)
        error = cudaMemcpy(staging_value.data(), device_candidate_value,
                           row_count(timed_tokens) * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        std::fclose(raw);
        return fail_cuda("stage candidate", error);
      }
    }
    const bool numeric_ok =
        results[c].vs_off.nonfinite == 0 &&
        (c == 0 ||
         (like_arithmetic(id)
              ? (results[c].outputs_byte_equal ||
                 (results[c].vs_off.max_abs == 0.0F &&
                  results[c].vs_off.rms == 0.0F))
              : (results[c].vs_off.max_abs <= kOpt044Abs &&
                 results[c].vs_off.rms <= kOpt044Rms)));
    results[c].eligible =
        results[c].launch_ok && results[c].occupancy >= 1 && numeric_ok &&
        results[c].scratch_unchanged && results[c].committed_unchanged &&
        results[c].candidate_exact && results[c].graph_eager_equal;
    std::fprintf(raw,
                 "pipeline_ab id=%s occupancy=%d launch_ok=%s "
                 "max_abs=%.9g rms=%.9g cosine=%.9g nonfinite=%zu scratch=%s "
                 "committed=%s candidate=%s byte_equal=%s graph_eager=%s "
                 "eligible=%s\n",
                 id, results[c].occupancy, json_bool(results[c].launch_ok),
                 static_cast<double>(results[c].vs_off.max_abs),
                 static_cast<double>(results[c].vs_off.rms),
                 static_cast<double>(results[c].vs_off.one_minus_cosine),
                 results[c].vs_off.nonfinite,
                 json_bool(results[c].scratch_unchanged),
                 json_bool(results[c].committed_unchanged),
                 json_bool(results[c].candidate_exact),
                 json_bool(results[c].outputs_byte_equal),
                 json_bool(results[c].graph_eager_equal),
                 json_bool(results[c].eligible));
  }
  if (!results[0].eligible) {
    std::fclose(raw);
    std::fprintf(stderr, "off candidate is not eligible\n");
    return 1;
  }

  struct CaseMetrics {
    std::size_t start_position = 0;
    std::size_t token_count = 0;
    bool launch_ok = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool eligible = false;
    Envelope vs_off;
  };
  std::vector<CaseMetrics> cases;
  bool all_eligible = true;
  const int start_limit =
      tier == qw38::cuda::TestTier::kSmoke ? 1 : kCorrectnessStartCount;
  const int token_limit =
      tier == qw38::cuda::TestTier::kSmoke ? 2 : kCorrectnessTokenCount;
  for (int start_index = 0; start_index < start_limit; ++start_index) {
    for (int token_index = 0; token_index < token_limit; ++token_index) {
      const std::size_t start = kCorrectnessStarts[start_index];
      const std::size_t tokens = kCorrectnessTokens[token_index];
      if (tokens > timed_tokens && tier == qw38::cuda::TestTier::kSmoke) continue;
      CaseMetrics item;
      item.start_position = start;
      item.token_count = tokens;
      if (start + tokens > kConfig.capacity) {
        item.eligible = false;
        all_eligible = false;
        cases.push_back(item);
        continue;
      }
      LaunchMetrics control;
      error = launch_one(start, tokens, "off", &control, nullptr, nullptr,
                         nullptr, false);
      bool case_ok = error == cudaSuccess && control.launch_ok;
      Envelope worst;
      bool committed = control.committed_unchanged;
      bool exact = true;
      if (case_ok) {
        std::vector<__nv_bfloat16> case_key(row_count(tokens));
        std::vector<__nv_bfloat16> case_value(row_count(tokens));
        error = cudaMemcpy(case_key.data(), device_candidate_key,
                           row_count(tokens) * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
        if (error == cudaSuccess)
          error = cudaMemcpy(case_value.data(), device_candidate_value,
                             row_count(tokens) * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost);
        for (int c = 1; c < kCandidateCount && error == cudaSuccess; ++c) {
          if (results[c].occupancy < 1) continue;
          LaunchMetrics cand;
          error = launch_one(start, tokens, kCandidates[c], &cand,
                             &control.output, &case_key, &case_value, false);
          const bool numeric =
              cand.vs_off.nonfinite == 0 &&
              (like_arithmetic(kCandidates[c])
                   ? (cand.outputs_byte_equal || cand.vs_off.max_abs == 0.0F)
                   : (cand.vs_off.max_abs <= kOpt044Abs &&
                      cand.vs_off.rms <= kOpt044Rms));
          case_ok = case_ok && cand.launch_ok && numeric &&
                    cand.scratch_unchanged && cand.committed_unchanged &&
                    cand.candidate_exact;
          committed = committed && cand.committed_unchanged;
          exact = exact && cand.candidate_exact;
          if (cand.vs_off.max_abs > worst.max_abs) worst = cand.vs_off;
        }
      }
      item.launch_ok = case_ok;
      item.committed_unchanged = committed;
      item.candidate_exact = exact;
      item.vs_off = worst;
      item.eligible = case_ok && error == cudaSuccess;
      all_eligible = all_eligible && item.eligible;
      cases.push_back(item);
      std::fprintf(raw,
                   "correctness start=%zu tokens=%zu launch_ok=%s eligible=%s "
                   "vs_off_max_abs=%.9g\n",
                   start, tokens, json_bool(item.launch_ok),
                   json_bool(item.eligible),
                   static_cast<double>(item.vs_off.max_abs));
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
  for (int round = 0; round < warmups + measured; ++round) {
    for (int c = 0; c < kCandidateCount; ++c) {
      qw38::cuda::QueryPreparePathScope prep("hoisted");
      qw38::cuda::AttentionPipelinePathScope pipe(kCandidates[c]);
      if (results[c].occupancy < 1 && c != 0) {
        if (round < warmups) {
          results[c].warmup_ms.push_back(0.0F);
        } else {
          results[c].samples.push_back(1.0e30F);
        }
        continue;
      }
      error = restore(timed_tokens, 0);
      if (error == cudaSuccess) error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_attention_prepare_chunk_stream_k(
            kConfig, 0, timed_tokens, device_query, device_key, device_value,
            device_query_scale, device_key_scale, device_gate, committed,
            candidate, device_normalized_query, device_normalized_key,
            device_scores, device_output, device_partial, device_meta, nullptr,
            device_prepared);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float ms = 0.0F;
      if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
      if (error != cudaSuccess) {
        std::fclose(raw);
        return fail_cuda("ab timed launch", error);
      }
      if (round < warmups) {
        results[c].warmup_ms.push_back(ms);
        std::fprintf(raw, "pipeline_ab id=%s warmup=%d ms=%.9g occupancy=%d\n",
                     kCandidates[c], round, static_cast<double>(ms),
                     results[c].occupancy);
      } else {
        results[c].samples.push_back(ms);
        std::fprintf(raw, "pipeline_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
                     kCandidates[c], round - warmups, static_cast<double>(ms),
                     results[c].occupancy);
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  for (int c = 0; c < kCandidateCount; ++c) {
    double sum = 0.0;
    for (float sample : results[c].samples) sum += sample;
    results[c].mean_ms = results[c].samples.empty()
                             ? 0.0F
                             : static_cast<float>(
                                   sum / static_cast<double>(results[c].samples.size()));
    std::fprintf(raw,
                 "pipeline_ab_mean id=%s mean_ms=%.9g occupancy=%d eligible=%s\n",
                 results[c].id, static_cast<double>(results[c].mean_ms),
                 results[c].occupancy, json_bool(results[c].eligible));
  }

  const char* model_path = argc > 2 ? argv[2] : nullptr;
  bool layers_ok = true;
  if (tier != qw38::cuda::TestTier::kSmoke && model_path != nullptr) {
    int fd = open(model_path, O_RDONLY);
    struct stat st {};
    void* map = MAP_FAILED;
    if (fd >= 0 && fstat(fd, &st) == 0)
      map = mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ,
                 MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
      layers_ok = false;
    } else {
      const char* base = static_cast<const char*>(map);
      const std::size_t layer_tokens =
          tier == qw38::cuda::TestTier::kAcceptance ? 64 : 16;
      for (int layer = 0; layer < kLayerCount; ++layer) {
        std::memcpy(query_scale.data(), base + kLayers[layer].q_off,
                    kConfig.head_width * sizeof(float));
        std::memcpy(key_scale.data(), base + kLayers[layer].k_off,
                    kConfig.head_width * sizeof(float));
        LaunchMetrics control;
        error = launch_one(0, layer_tokens, "off", &control, nullptr, nullptr,
                           nullptr, false);
        bool ok = error == cudaSuccess && control.launch_ok;
        float worst = 0.0F;
        for (int c = 1; c < kCandidateCount && error == cudaSuccess; ++c) {
          if (results[c].occupancy < 1) continue;
          LaunchMetrics cand;
          error = launch_one(0, layer_tokens, kCandidates[c], &cand,
                             &control.output, nullptr, nullptr, false);
          const bool numeric =
              cand.vs_off.nonfinite == 0 &&
              (like_arithmetic(kCandidates[c])
                   ? (cand.outputs_byte_equal || cand.vs_off.max_abs == 0.0F)
                   : cand.vs_off.max_abs <= kOpt044Abs);
          ok = ok && cand.launch_ok && numeric;
          worst = std::max(worst, cand.vs_off.max_abs);
        }
        layers_ok = layers_ok && ok;
        std::fprintf(raw, "layer id=%s tokens=%zu ok=%s max_abs=%.9g\n",
                     kLayers[layer].id, layer_tokens, json_bool(ok),
                     static_cast<double>(worst));
      }
      munmap(map, static_cast<std::size_t>(st.st_size));
    }
    if (fd >= 0) close(fd);
  }

  auto mean_of = [&](const char* path) -> float {
    return results[index_of(path)].mean_ms;
  };
  auto eligible_of = [&](const char* path) -> bool {
    return results[index_of(path)].eligible;
  };
  const float off_mean = mean_of("off");
  const bool admit_reg =
      all_eligible && layers_ok && eligible_of("dual_reg") &&
      mean_of("dual_reg") < off_mean;
  const bool admit_f16_shared =
      all_eligible && layers_ok && eligible_of("f16") && mean_of("f16") < off_mean;
  const bool admit_f16_on_reg =
      all_eligible && layers_ok && eligible_of("f16_reg") &&
      eligible_of("dual_reg") && mean_of("f16_reg") < mean_of("dual_reg");
  const bool admit_f16 = admit_f16_shared || admit_f16_on_reg ||
                         (eligible_of("f16_reg") && !admit_reg &&
                          all_eligible && layers_ok &&
                          mean_of("f16_reg") < off_mean);
  const char* arithmetic = "off";
  float arithmetic_mean = off_mean;
  if (admit_reg && admit_f16 && eligible_of("f16_reg")) {
    arithmetic = "f16_reg";
    arithmetic_mean = mean_of("f16_reg");
  } else if (admit_reg) {
    arithmetic = "dual_reg";
    arithmetic_mean = mean_of("dual_reg");
  } else if (admit_f16 && admit_f16_shared) {
    arithmetic = "f16";
    arithmetic_mean = mean_of("f16");
  } else if (admit_f16 && eligible_of("f16_reg")) {
    arithmetic = "f16_reg";
    arithmetic_mean = mean_of("f16_reg");
  }

  const char* winner = arithmetic;
  float winner_mean = arithmetic_mean;
  const char* async_id = admit_f16 ? "f16_async" : "dual_async";
  if ((std::strcmp(arithmetic, "f16_reg") == 0 ||
       std::strcmp(arithmetic, "dual_reg") == 0) &&
      eligible_of(async_id) && mean_of(async_id) < winner_mean) {
    winner = async_id;
    winner_mean = mean_of(async_id);
  }
  if (admit_f16 && eligible_of("nbatch64") && mean_of("nbatch64") < winner_mean) {
    winner = "nbatch64";
    winner_mean = mean_of("nbatch64");
  }
  if (admit_f16 && eligible_of("gqa6") && mean_of("gqa6") < winner_mean) {
    winner = "gqa6";
    winner_mean = mean_of("gqa6");
  }
  const bool win = std::strcmp(winner, "off") != 0;
  std::fprintf(raw,
               "pipeline_ab_winner id=%s win=%s mean_ms=%.9g admit_f16=%s "
               "admit_reg=%s arithmetic=%s\n",
               winner, json_bool(win), static_cast<double>(winner_mean),
               json_bool(admit_f16), json_bool(admit_reg), arithmetic);

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  char utc[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));

  std::printf("%s{", kPrefix);
  std::printf("\"schema_version\":1,\"task\":\"OPT-051\",\"status\":\"measured\",");
  std::printf("\"measurement_utc\":\"%s\",\"device\":\"%s\",", utc, prop.name);
  std::printf("\"compute_capability\":\"%d.%d\",", prop.major, prop.minor);
  std::printf("\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\",", kLlamaRev,
              kGgufSha);
  std::printf("\"tier\":\"%s\",\"timed_tokens\":%zu,", qw38::cuda::test_tier_name(),
              timed_tokens);
  std::printf("\"selected_attention_pipeline_path\":\"%s\",", winner);
  std::printf("\"selected_query_prepare_path\":\"hoisted\",");
  std::printf("\"admit_f16\":%s,\"admit_reg\":%s,\"arithmetic\":\"%s\",",
              json_bool(admit_f16), json_bool(admit_reg), arithmetic);
  std::printf("\"ab\":{\"winner\":\"%s\",\"win\":%s,\"candidates\":{", winner,
              json_bool(win));
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::printf(",");
    std::printf("\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"occupancy\":%d,",
                results[c].id, results[c].id, results[c].id,
                results[c].occupancy);
    std::printf("\"launch_ok\":%s,\"eligible\":%s,",
                json_bool(results[c].launch_ok), json_bool(results[c].eligible));
    std::printf("\"outputs_byte_equal\":%s,\"graph_eager_equal\":%s,",
                json_bool(results[c].outputs_byte_equal),
                json_bool(results[c].graph_eager_equal));
    std::printf("\"candidate_exact\":%s,\"committed_unchanged\":%s,",
                json_bool(results[c].candidate_exact),
                json_bool(results[c].committed_unchanged));
    std::printf("\"scratch_unchanged\":%s,\"mean_ms\":%.9g,",
                json_bool(results[c].scratch_unchanged),
                static_cast<double>(results[c].mean_ms));
    print_array(stdout, "warmup_ms", results[c].warmup_ms);
    std::printf(",");
    print_array(stdout, "samples", results[c].samples);
    std::printf(",\"vs_off\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                "\"one_minus_cosine\":%.9g,\"nonfinite\":%zu}}",
                static_cast<double>(results[c].vs_off.max_abs),
                static_cast<double>(results[c].vs_off.rms),
                static_cast<double>(results[c].vs_off.one_minus_cosine),
                results[c].vs_off.nonfinite);
  }
  std::printf("}},\"correctness\":{\"all_eligible\":%s,\"layers_ok\":%s,"
              "\"case_count\":%zu},\"cp_async_source\":true}\n",
              json_bool(all_eligible), json_bool(layers_ok), cases.size());
  std::fprintf(stderr, "status=passed winner=%s\n", winner);

  cudaFree(device_query);
  cudaFree(device_key);
  cudaFree(device_value);
  cudaFree(device_gate);
  cudaFree(device_query_scale);
  cudaFree(device_key_scale);
  cudaFree(device_normalized_query);
  cudaFree(device_normalized_key);
  cudaFree(device_scores);
  cudaFree(device_output);
  cudaFree(device_partial);
  cudaFree(device_meta);
  cudaFree(device_prepared);
  cudaFree(device_committed_key);
  cudaFree(device_committed_value);
  cudaFree(device_candidate_key);
  cudaFree(device_candidate_value);
  std::fclose(raw);
  return 0;
}
