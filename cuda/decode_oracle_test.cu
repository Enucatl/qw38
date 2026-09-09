#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "full_scheduler.h"
#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr std::size_t kCapacity = 131072;
constexpr std::size_t kDecodeTokens = 256;
constexpr int kWarmups = 3;
constexpr int kRuns = 30;
constexpr char kPrefix[] = "QW38_DECODE_ORACLE_RESULT=";

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

float percentile(std::vector<float> values, float fraction) {
  if (values.empty()) return 0.0F;
  std::sort(values.begin(), values.end());
  const float position = fraction * static_cast<float>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const float weight = position - static_cast<float>(lower);
  return values[lower] * (1.0F - weight) + values[upper] * weight;
}

void print_array(const char* key, const std::vector<float>& values) {
  std::printf("\"%s\":[", key);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::printf(",");
    std::printf("%.9g", static_cast<double>(values[index]));
  }
  std::printf("]");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf PREFIX\n", argv[0]);
    return 2;
  }
  const std::size_t prefix = static_cast<std::size_t>(std::atoi(argv[2]));
  if (prefix != 128 && prefix != 2048) {
    std::fprintf(stderr, "PREFIX must be 128 or 2048\n");
    return 2;
  }

  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(argv[1], &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(argv[1]);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  qw38::cuda::ResidentModel model;
  if (status.is_ok()) {
    status = model.upload(weights, mapping.data(), mapping.size());
  }
  if (!status.is_ok()) return fail_status(status);

  std::vector<std::size_t> tokens(prefix + kDecodeTokens);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
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

  const int total_runs = kWarmups + kRuns;
  std::vector<float> warmup_tok_s(kWarmups);
  std::vector<float> tok_s(kRuns);
  std::vector<float> run_wall_ms(kRuns);
  std::vector<float> token_latency_ms;
  token_latency_ms.reserve(static_cast<std::size_t>(kRuns) * kDecodeTokens);
  std::vector<float> run_mean_token_latency_ms(kRuns);

  for (int run = 0; run < total_runs; ++run) {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    status = session.create(kCapacity);
    if (status.is_ok()) status = workspace.create(kCapacity);
    if (status.is_ok()) status = graphs.create(model, &workspace);
    if (!status.is_ok()) return fail_status(status);
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    qw38::cuda::SyncResult result{};
    status = qw38::cuda::sync_tokens(
        model, tokens.data(), prefix, &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &result, nullptr, &graphs);
    if (!status.is_ok()) return fail_status(status);

    std::vector<float> run_tokens(kDecodeTokens);
    const auto decode_started = std::chrono::steady_clock::now();
    for (std::size_t step = 0; step < kDecodeTokens; ++step) {
      float elapsed_ms = 0.0F;
      const auto token_started = std::chrono::steady_clock::now();
      status = qw38::cuda::execute_token(
          model, tokens[prefix + step], &session, &workspace, logits.data(),
          logits.size(), hidden.data(), hidden.size(), &elapsed_ms, nullptr,
          nullptr, qw38::cuda::PointwisePath::kFused, &graphs, nullptr);
      if (!status.is_ok()) return fail_status(status);
      run_tokens[step] = static_cast<float>(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - token_started)
              .count());
    }
    const float wall_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decode_started)
            .count());
    const float run_tok_s =
        wall_ms > 0.0F
            ? static_cast<float>(kDecodeTokens) * 1000.0F / wall_ms
            : 0.0F;
    if (run < kWarmups) {
      warmup_tok_s[run] = run_tok_s;
    } else {
      const int measured = run - kWarmups;
      tok_s[measured] = run_tok_s;
      run_wall_ms[measured] = wall_ms;
      float mean_token = 0.0F;
      for (float value : run_tokens) mean_token += value;
      mean_token /= static_cast<float>(kDecodeTokens);
      run_mean_token_latency_ms[measured] = mean_token;
      token_latency_ms.insert(token_latency_ms.end(), run_tokens.begin(),
                              run_tokens.end());
    }
  }

  float mean_tok_s = 0.0F;
  for (float value : tok_s) mean_tok_s += value;
  mean_tok_s /= static_cast<float>(kRuns);
  const float p50 = percentile(token_latency_ms, 0.50F);
  const float p95 = percentile(token_latency_ms, 0.95F);
  const float run_mean_p95 = percentile(run_mean_token_latency_ms, 0.95F);

  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-032\",\"status\":\"measured\",",
              kPrefix);
  std::printf("\"device\":\"%s\",\"compute_capability\":\"%d.%d\",", prop.name,
              prop.major, prop.minor);
  std::printf("\"measurement_utc\":\"%s\",\"prefix\":%zu,\"decode_tokens\":%zu,",
              utc, prefix, kDecodeTokens);
  std::printf("\"warmups\":%d,\"runs\":%d,\"mean_tok_s\":%.9g,", kWarmups, kRuns,
              static_cast<double>(mean_tok_s));
  std::printf("\"token_latency_p50_ms\":%.9g,\"token_latency_p95_ms\":%.9g,",
              static_cast<double>(p50), static_cast<double>(p95));
  std::printf("\"run_mean_token_latency_p95_ms\":%.9g,",
              static_cast<double>(run_mean_p95));
  std::printf("\"graphs_created\":true,\"attribution\":null,\"cache_policy\":\"disabled\",");
  print_array("warmup_tok_s", warmup_tok_s);
  std::printf(",");
  print_array("tok_s", tok_s);
  std::printf(",");
  print_array("run_wall_ms", run_wall_ms);
  std::printf(",");
  print_array("token_latency_ms", token_latency_ms);
  std::printf(",\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\"}\n");
  std::printf("status=passed\n");
  return 0;
}
