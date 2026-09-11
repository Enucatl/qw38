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

constexpr char kPrefix[] = "QW38_OPT050_ATTENTION_QUERY_PREPARE_AB_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr const char* kCandidates[] = {"in_kernel", "hoisted", "hoisted_fma"};
constexpr int kCandidateCount = 3;
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
  if (std::strcmp(path, "in_kernel") == 0)
    return qw38::cuda::fattn_warp_qk_occupancy();
  const int prep = qw38::cuda::attention_prepare_query_occupancy();
  const int attn = qw38::cuda::fattn_prepared_query_occupancy();
  return prep < attn ? prep : attn;
}

bool like_arithmetic(const char* path) {
  return std::strcmp(path, "hoisted") == 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_legacy_ok()) {
    std::fprintf(stderr, "%s\n", qw38::cuda::test_tier_legacy_error());
    return 1;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  const int warmups = qw38::cuda::test_warmups();
  const int measured = qw38::cuda::test_samples();
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt050-attention-query-prepare/"
                 "attention-query-prepare-ab-raw.txt";
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
    Envelope vs_in_kernel;
    Envelope vs_prepared;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool outputs_byte_equal = false;
    bool prepared_q_equal = false;
    bool graph_eager_equal = false;
    bool launch_ok = false;
    std::vector<float> output;
    std::vector<__half> prepared;
  };

  auto launch_one = [&](std::size_t start_position, std::size_t token_count,
                        const char* path, LaunchMetrics* metrics,
                        const std::vector<float>* baseline,
                        const std::vector<__nv_bfloat16>* staging_key,
                        const std::vector<__nv_bfloat16>* staging_value,
                        bool capture_graph) -> cudaError_t {
    qw38::cuda::QueryPreparePathScope scope(path);
    cudaError_t local = restore(token_count, start_position);
    const __half* prepared =
        qw38::cuda::query_prepare_is_hoisted(path) ? device_prepared
                                                         : nullptr;
    auto run = [&]() -> cudaError_t {
      return qw38::cuda::launch_attention_prepare_chunk_stream_k(
          kConfig, start_position, token_count, device_query, device_key,
          device_value, device_query_scale, device_key_scale, device_gate,
          committed, candidate, device_normalized_query, device_normalized_key,
          device_scores, device_output, device_partial, device_meta, nullptr,
          prepared);
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
              device_partial, device_meta, stream, prepared);
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
    if (qw38::cuda::query_prepare_is_hoisted(path) && token_count >= 16) {
      const std::size_t halfs =
          qw38::cuda::attention_prepared_query_bytes(kConfig, token_count) /
          sizeof(__half);
      metrics->prepared.assign(halfs, __float2half_rn(0.0F));
      local = cudaMemcpy(metrics->prepared.data(), device_prepared,
                         halfs * sizeof(__half), cudaMemcpyDeviceToHost);
      if (local != cudaSuccess) return local;
      metrics->prepared_q_equal = true;
      for (std::size_t index = 0; index < halfs; ++index) {
        if (!std::isfinite(__half2float(metrics->prepared[index]))) {
          metrics->prepared_q_equal = false;
          break;
        }
      }
    } else {
      metrics->prepared_q_equal = true;
    }
    if (baseline != nullptr) {
      metrics->vs_in_kernel = compare(metrics->output, *baseline);
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
    const char* id = "in_kernel";
    int occupancy = 0;
    Envelope vs_in_kernel;
    bool scratch_unchanged = false;
    bool committed_unchanged = false;
    bool candidate_exact = false;
    bool outputs_byte_equal = false;
    bool prepared_q_equal = false;
    bool graph_eager_equal = false;
    bool launch_ok = false;
    bool eligible = false;
    std::vector<float> warmup_ms;
    std::vector<float> samples;
    float mean_ms = 0.0F;
    std::vector<float> output;
  };
  CandidateResult results[kCandidateCount];
  std::vector<float> in_kernel_output;
  std::vector<__nv_bfloat16> staging_key;
  std::vector<__nv_bfloat16> staging_value;

  for (int c = 0; c < kCandidateCount; ++c) {
    const char* id = kCandidates[c];
    results[c].id = id;
    results[c].occupancy = occupancy_for(id);
    LaunchMetrics metrics;
    const std::vector<float>* baseline = c == 0 ? nullptr : &in_kernel_output;
    const std::vector<__nv_bfloat16>* sk = c == 0 ? nullptr : &staging_key;
    const std::vector<__nv_bfloat16>* sv = c == 0 ? nullptr : &staging_value;
    error = launch_one(0, timed_tokens, id, &metrics, baseline, sk, sv,
                       c != 0 && timed_tokens >= 16);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("ab candidate launch", error);
    }
    results[c].launch_ok = metrics.launch_ok;
    results[c].output = metrics.output;
    results[c].vs_in_kernel = metrics.vs_in_kernel;
    results[c].scratch_unchanged = metrics.scratch_unchanged;
    results[c].committed_unchanged = metrics.committed_unchanged;
    results[c].candidate_exact = metrics.candidate_exact;
    results[c].outputs_byte_equal = metrics.outputs_byte_equal;
    results[c].prepared_q_equal = metrics.prepared_q_equal;
    results[c].graph_eager_equal = metrics.graph_eager_equal;
    if (c == 0) {
      in_kernel_output = metrics.output;
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
        results[c].vs_in_kernel.nonfinite == 0 &&
        (c == 0 ||
         (like_arithmetic(id)
              ? (results[c].outputs_byte_equal ||
                 (results[c].vs_in_kernel.max_abs == 0.0F &&
                  results[c].vs_in_kernel.rms == 0.0F))
              : (results[c].vs_in_kernel.max_abs <= kOpt044Abs &&
                 results[c].vs_in_kernel.rms <= kOpt044Rms)));
    results[c].eligible =
        results[c].launch_ok && results[c].occupancy >= 1 && numeric_ok &&
        results[c].scratch_unchanged && results[c].committed_unchanged &&
        results[c].candidate_exact && results[c].prepared_q_equal &&
        results[c].graph_eager_equal;
    std::fprintf(raw,
                 "query_prep_ab id=%s occupancy=%d launch_ok=%s "
                 "max_abs=%.9g rms=%.9g cosine=%.9g nonfinite=%zu scratch=%s "
                 "committed=%s candidate=%s byte_equal=%s prepared_q=%s "
                 "graph_eager=%s eligible=%s\n",
                 id, results[c].occupancy, json_bool(results[c].launch_ok),
                 static_cast<double>(results[c].vs_in_kernel.max_abs),
                 static_cast<double>(results[c].vs_in_kernel.rms),
                 static_cast<double>(results[c].vs_in_kernel.one_minus_cosine),
                 results[c].vs_in_kernel.nonfinite,
                 json_bool(results[c].scratch_unchanged),
                 json_bool(results[c].committed_unchanged),
                 json_bool(results[c].candidate_exact),
                 json_bool(results[c].outputs_byte_equal),
                 json_bool(results[c].prepared_q_equal),
                 json_bool(results[c].graph_eager_equal),
                 json_bool(results[c].eligible));
  }
  if (!results[0].eligible) {
    std::fclose(raw);
    std::fprintf(stderr, "in_kernel candidate is not eligible\n");
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
    Envelope vs_in_kernel;
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
      error = launch_one(start, tokens, "in_kernel", &control, nullptr, nullptr,
                         nullptr, false);
      LaunchMetrics hoisted;
      LaunchMetrics fma;
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
          error = launch_one(start, tokens, "hoisted", &hoisted, &control.output,
                             &case_key, &case_value, false);
        if (error == cudaSuccess)
          error = launch_one(start, tokens, "hoisted_fma", &fma, &control.output,
                             &case_key, &case_value, false);
      }
      item.launch_ok = error == cudaSuccess && control.launch_ok &&
                       hoisted.launch_ok && fma.launch_ok;
      item.outputs_byte_equal = hoisted.outputs_byte_equal;
      item.scratch_unchanged = control.scratch_unchanged &&
                               hoisted.scratch_unchanged && fma.scratch_unchanged;
      item.committed_unchanged =
          control.committed_unchanged && hoisted.committed_unchanged &&
          fma.committed_unchanged;
      item.candidate_exact = hoisted.candidate_exact && fma.candidate_exact;
      item.vs_in_kernel = hoisted.vs_in_kernel;
      const bool fma_ok = fma.vs_in_kernel.nonfinite == 0 &&
                          fma.vs_in_kernel.max_abs <= kOpt044Abs &&
                          fma.vs_in_kernel.rms <= kOpt044Rms;
      item.eligible = item.launch_ok && item.outputs_byte_equal &&
                      item.scratch_unchanged && item.committed_unchanged &&
                      item.candidate_exact && fma_ok &&
                      item.vs_in_kernel.nonfinite == 0;
      all_eligible = all_eligible && item.eligible;
      cases.push_back(item);
      std::fprintf(raw,
                   "correctness start=%zu tokens=%zu launch_ok=%s byte_equal=%s "
                   "eligible=%s vs_in_kernel_max_abs=%.9g fma_max_abs=%.9g\n",
                   start, tokens, json_bool(item.launch_ok),
                   json_bool(item.outputs_byte_equal), json_bool(item.eligible),
                   static_cast<double>(item.vs_in_kernel.max_abs),
                   static_cast<double>(fma.vs_in_kernel.max_abs));
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
      qw38::cuda::QueryPreparePathScope scope(kCandidates[c]);
      error = restore(timed_tokens, 0);
      const __half* prepared =
          qw38::cuda::query_prepare_is_hoisted(kCandidates[c])
              ? device_prepared
              : nullptr;
      if (error == cudaSuccess) error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_attention_prepare_chunk_stream_k(
            kConfig, 0, timed_tokens, device_query, device_key, device_value,
            device_query_scale, device_key_scale, device_gate, committed,
            candidate, device_normalized_query, device_normalized_key,
            device_scores, device_output, device_partial, device_meta, nullptr,
            prepared);
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
        std::fprintf(raw, "query_prep_ab id=%s warmup=%d ms=%.9g occupancy=%d\n",
                     kCandidates[c], round, static_cast<double>(ms),
                     results[c].occupancy);
      } else {
        results[c].samples.push_back(ms);
        std::fprintf(raw, "query_prep_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
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
                 "query_prep_ab_mean id=%s mean_ms=%.9g occupancy=%d eligible=%s\n",
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
        error = launch_one(0, layer_tokens, "in_kernel", &control, nullptr,
                           nullptr, nullptr, false);
        LaunchMetrics hoisted;
        if (error == cudaSuccess)
          error = launch_one(0, layer_tokens, "hoisted", &hoisted,
                             &control.output, nullptr, nullptr, false);
        LaunchMetrics fma;
        if (error == cudaSuccess)
          error = launch_one(0, layer_tokens, "hoisted_fma", &fma,
                             &control.output, nullptr, nullptr, false);
        const bool ok =
            error == cudaSuccess && hoisted.outputs_byte_equal &&
            fma.vs_in_kernel.nonfinite == 0 &&
            fma.vs_in_kernel.max_abs <= kOpt044Abs;
        layers_ok = layers_ok && ok;
        std::fprintf(raw,
                     "layer id=%s tokens=%zu ok=%s hoisted_max_abs=%.9g "
                     "fma_max_abs=%.9g\n",
                     kLayers[layer].id, layer_tokens, json_bool(ok),
                     static_cast<double>(hoisted.vs_in_kernel.max_abs),
                     static_cast<double>(fma.vs_in_kernel.max_abs));
      }
      munmap(map, static_cast<std::size_t>(st.st_size));
    }
    if (fd >= 0) close(fd);
  }

  const float baseline_mean = results[0].mean_ms;
  const char* winner = "in_kernel";
  float winner_mean = baseline_mean;
  if (all_eligible && results[1].eligible && layers_ok &&
      results[1].mean_ms < baseline_mean) {
    winner = "hoisted";
    winner_mean = results[1].mean_ms;
    if (results[2].eligible &&
        results[2].mean_ms < results[1].mean_ms * 0.99F) {
      winner = "hoisted_fma";
      winner_mean = results[2].mean_ms;
    }
  }
  const bool win = std::strcmp(winner, "in_kernel") != 0;
  std::fprintf(raw, "query_prep_ab_winner id=%s win=%s mean_ms=%.9g\n", winner,
               json_bool(win), static_cast<double>(winner_mean));

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  char utc[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));

  std::printf("%s{", kPrefix);
  std::printf("\"schema_version\":1,\"task\":\"OPT-050\",\"status\":\"measured\",");
  std::printf("\"measurement_utc\":\"%s\",\"device\":\"%s\",", utc, prop.name);
  std::printf("\"compute_capability\":\"%d.%d\",", prop.major, prop.minor);
  std::printf("\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\",", kLlamaRev,
              kGgufSha);
  std::printf("\"tier\":\"%s\",\"timed_tokens\":%zu,", qw38::cuda::test_tier_name(),
              timed_tokens);
  std::printf("\"selected_query_prepare_path\":\"%s\",", winner);
  std::printf("\"opt046_selected_q4_decode_path\":\"packed\",");
  std::printf("\"opt047_selected_q8_decode_path\":\"dp4a_q8_1\",");
  std::printf("\"opt048_selected_q6_decode_path\":\"integer_q8_1\",");
  std::printf("\"opt049_selected_ffn_decode_path\":\"paired_staged\",");
  std::printf("\"ab\":{\"winner\":\"%s\",\"win\":%s,\"candidates\":{", winner,
              json_bool(win));
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::printf(",");
    std::printf("\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"occupancy\":%d,",
                results[c].id, results[c].id, results[c].id,
                results[c].occupancy);
    std::printf("\"launch_ok\":%s,\"eligible\":%s,",
                json_bool(results[c].launch_ok), json_bool(results[c].eligible));
    std::printf("\"outputs_byte_equal\":%s,\"prepared_q_equal\":%s,",
                json_bool(results[c].outputs_byte_equal),
                json_bool(results[c].prepared_q_equal));
    std::printf("\"graph_eager_equal\":%s,\"candidate_exact\":%s,",
                json_bool(results[c].graph_eager_equal),
                json_bool(results[c].candidate_exact));
    std::printf("\"committed_unchanged\":%s,\"scratch_unchanged\":%s,",
                json_bool(results[c].committed_unchanged),
                json_bool(results[c].scratch_unchanged));
    std::printf("\"mean_ms\":%.9g,", static_cast<double>(results[c].mean_ms));
    print_array(stdout, "warmup_ms", results[c].warmup_ms);
    std::printf(",");
    print_array(stdout, "samples", results[c].samples);
    std::printf(",\"vs_in_kernel\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                "\"one_minus_cosine\":%.9g,\"nonfinite\":%zu}}",
                static_cast<double>(results[c].vs_in_kernel.max_abs),
                static_cast<double>(results[c].vs_in_kernel.rms),
                static_cast<double>(results[c].vs_in_kernel.one_minus_cosine),
                results[c].vs_in_kernel.nonfinite);
  }
  std::printf("}},\"correctness\":{\"all_eligible\":%s,\"layers_ok\":%s,"
              "\"case_count\":%zu},",
              json_bool(all_eligible), json_bool(layers_ok), cases.size());
  std::printf("\"prepared_bytes\":%zu,\"overlay_alias\":\"prompt_projection_a_\"}\n",
              qw38::cuda::attention_prepared_query_bytes(kConfig, timed_tokens));
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
