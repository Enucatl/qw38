#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kWarmups = 3;
constexpr int kRuns = 30;
constexpr int kDecodeTokens = 256;
constexpr int kVocabulary = 248320;
constexpr char kPrefix[] = "QW38_LLAMA_DECODE_ORACLE_RESULT=";

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void print_array(const char* key, const std::vector<double>& values) {
  std::printf("\"%s\":[", key);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::printf(",");
    std::printf("%.9g", values[index]);
  }
  std::printf("]");
}

std::int32_t token_at(std::size_t index) {
  return static_cast<std::int32_t>((42 + index * 997) % kVocabulary);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf PREFIX\n", argv[0]);
    return 2;
  }
  const int prefix = std::atoi(argv[2]);
  if (prefix != 128 && prefix != 2048) {
    std::fprintf(stderr, "PREFIX must be 128 or 2048\n");
    return 2;
  }

  llama_backend_init();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 99;
  llama_model* model = llama_model_load_from_file(argv[1], model_params);
  if (model == nullptr) {
    llama_backend_free();
    return 1;
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = 4096;
  context_params.n_batch = 2048;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.no_perf = true;

  std::vector<llama_token> tokens(static_cast<std::size_t>(prefix) +
                                  kDecodeTokens);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = token_at(index);
  }

  const int total_runs = kWarmups + kRuns;
  std::vector<double> warmup_tok_s(kWarmups);
  std::vector<double> tok_s(kRuns);
  std::vector<double> run_wall_ms(kRuns);
  std::vector<double> token_latency_ms;
  token_latency_ms.reserve(static_cast<std::size_t>(kRuns) * kDecodeTokens);
  std::vector<double> run_mean_token_latency_ms(kRuns);

  for (int run = 0; run < total_runs; ++run) {
    llama_context* ctx = llama_init_from_model(model, context_params);
    if (ctx == nullptr) {
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    llama_batch prefix_batch =
        llama_batch_get_one(tokens.data(), prefix);
    if (llama_decode(ctx, prefix_batch) != 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }

    std::vector<double> run_tokens(kDecodeTokens);
    const int64_t decode_started = llama_time_us();
    for (int step = 0; step < kDecodeTokens; ++step) {
      llama_token token = tokens[static_cast<std::size_t>(prefix + step)];
      const int64_t token_started = llama_time_us();
      llama_batch batch = llama_batch_get_one(&token, 1);
      if (llama_decode(ctx, batch) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
      }
      const float* logits = llama_get_logits_ith(ctx, -1);
      if (logits == nullptr) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
      }
      volatile float sink = logits[0];
      (void)sink;
      run_tokens[static_cast<std::size_t>(step)] =
          static_cast<double>(llama_time_us() - token_started) / 1000.0;
    }
    const double wall_ms =
        static_cast<double>(llama_time_us() - decode_started) / 1000.0;
    const double run_tok_s =
        wall_ms > 0.0 ? static_cast<double>(kDecodeTokens) * 1000.0 / wall_ms
                      : 0.0;
    llama_free(ctx);
    if (run < kWarmups) {
      warmup_tok_s[run] = run_tok_s;
    } else {
      const int measured = run - kWarmups;
      tok_s[measured] = run_tok_s;
      run_wall_ms[measured] = wall_ms;
      double mean_token = 0.0;
      for (double value : run_tokens) mean_token += value;
      mean_token /= static_cast<double>(kDecodeTokens);
      run_mean_token_latency_ms[measured] = mean_token;
      token_latency_ms.insert(token_latency_ms.end(), run_tokens.begin(),
                              run_tokens.end());
    }
  }

  double mean_tok_s = 0.0;
  for (double value : tok_s) mean_tok_s += value;
  mean_tok_s /= static_cast<double>(kRuns);
  const double p50 = percentile(token_latency_ms, 0.50);
  const double p95 = percentile(token_latency_ms, 0.95);
  const double run_mean_p95 = percentile(run_mean_token_latency_ms, 0.95);

  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-032\",\"status\":\"measured\",",
              kPrefix);
  std::printf("\"prefix\":%d,\"decode_tokens\":%d,\"warmups\":%d,\"runs\":%d,",
              prefix, kDecodeTokens, kWarmups, kRuns);
  std::printf("\"n_gpu_layers\":99,\"n_ctx\":4096,\"n_batch\":2048,\"n_ubatch\":512,");
  std::printf("\"mean_tok_s\":%.9g,\"token_latency_p50_ms\":%.9g,", mean_tok_s,
              p50);
  std::printf("\"token_latency_p95_ms\":%.9g,\"run_mean_token_latency_p95_ms\":%.9g,",
              p95, run_mean_p95);
  print_array("warmup_tok_s", warmup_tok_s);
  std::printf(",");
  print_array("tok_s", tok_s);
  std::printf(",");
  print_array("run_wall_ms", run_wall_ms);
  std::printf(",");
  print_array("token_latency_ms", token_latency_ms);
  std::printf("}\n");

  llama_model_free(model);
  llama_backend_free();
  return 0;
}
