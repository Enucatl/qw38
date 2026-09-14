#include "llama.h"

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kVocabulary = 248320;
constexpr int kDefaultTokens = 256;
constexpr int kDefaultCtx = 131072;
constexpr int kWindowTokens = 12;
constexpr int kWindowStarts[] = {0, 122, 244};
constexpr char kPrefix[] = "QW38_OPT136_GRAPH_ACCOUNTING_RESULT=";
constexpr char kRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";

struct Options final {
  const char* model = nullptr;
  const char* workload = "unprofiled";
  int prefix = 128;
  int tokens = kDefaultTokens;
  int ctx = kDefaultCtx;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

std::int32_t token_at(std::size_t index) {
  return static_cast<std::int32_t>((42 + index * 997) % kVocabulary);
}

bool legal_prefix(int prefix) {
  return prefix == 128 || prefix == 2048 || prefix == 8192 || prefix == 32768;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s MODEL.gguf [--workload identity|unprofiled|"
               "graph-capture|node-capture] [--prefix N] [--tokens N] "
               "[--ctx N]\n",
               argv0);
  return 2;
}

bool parse_int(const char* text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->prefix)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--tokens") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->tokens)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--ctx") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->ctx)) return usage(argv[0]);
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    } else if (options->model == nullptr) {
      options->model = arg;
    } else {
      return usage(argv[0]);
    }
  }
  if (options->model == nullptr) return usage(argv[0]);
  if (!legal_prefix(options->prefix)) {
    std::fprintf(stderr, "PREFIX must be 128, 2048, 8192, or 32768\n");
    return 2;
  }
  return 0;
}

bool in_window(int step) {
  for (int start : kWindowStarts) {
    if (step >= start && step < start + kWindowTokens) return true;
  }
  return false;
}

bool window_begin(int step) {
  for (int start : kWindowStarts) {
    if (step == start) return true;
  }
  return false;
}

bool window_end(int step) {
  for (int start : kWindowStarts) {
    if (step + 1 == start + kWindowTokens) return true;
  }
  return false;
}

double wall_ms(std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - started)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  const int parsed = parse_args(argc, argv, &options);
  if (parsed != 0) return parsed;

  llama_backend_init();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 99;
  llama_model* model = llama_model_load_from_file(options.model, model_params);
  if (model == nullptr) {
    llama_backend_free();
    return 1;
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = options.ctx;
  context_params.n_batch = 2048;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.no_perf = true;

  std::vector<llama_token> tokens(
      static_cast<std::size_t>(options.prefix + options.tokens));
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = token_at(index);
  }

  llama_context* ctx = llama_init_from_model(model, context_params);
  if (ctx == nullptr) {
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  const bool capture = std::strcmp(options.workload, "graph-capture") == 0 ||
                       std::strcmp(options.workload, "node-capture") == 0;
  const auto prefill_started = std::chrono::steady_clock::now();
  int offset = 0;
  while (offset < options.prefix) {
    const int n = std::min(static_cast<int>(context_params.n_batch),
                           options.prefix - offset);
    llama_batch prefix_batch =
        llama_batch_get_one(tokens.data() + offset, n);
    if (llama_decode(ctx, prefix_batch) != 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    offset += n;
  }
  const double prefill_ms = wall_ms(prefill_started);

  if (std::strcmp(options.workload, "identity") == 0) {
    std::printf(
        "%s{\"schema_version\":1,\"task\":\"OPT-136\",\"workload\":\"identity\","
        "\"ok\":true,\"engine\":\"llama\",\"llama_revision\":\"%s\","
        "\"prefix\":%d,\"capacity\":%d,\"keep\":false}\n",
        kPrefix, kRevision, options.prefix, options.ctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
  }

  const auto decode_started = std::chrono::steady_clock::now();
  int window_steps = 0;
  bool profiler_started = false;
  for (int step = 0; step < options.tokens; ++step) {
    const bool capture_step = capture && in_window(step);
    if (capture_step && window_begin(step)) {
      nvtxRangePushA("opt136.window");
      cudaProfilerStart();
      profiler_started = true;
    }
    char label[192];
    std::snprintf(label, sizeof(label),
                  "opt136 engine=llama prefix=%d step=%d layer=-1 op=eval",
                  options.prefix, step);
    nvtxRangePushA(label);
    llama_token token = tokens[static_cast<std::size_t>(options.prefix + step)];
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch) != 0) {
      nvtxRangePop();
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    const float* logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
      nvtxRangePop();
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    volatile float sink = logits[0];
    (void)sink;
    nvtxRangePop();
    if (capture_step) ++window_steps;
    if (capture_step && window_end(step) && profiler_started) {
      cudaProfilerStop();
      nvtxRangePop();
      profiler_started = false;
    }
  }
  const double decode_only_ms = wall_ms(decode_started);
  const double complete_request_ms = prefill_ms + decode_only_ms;

  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-136\",\"workload\":\"%s\","
      "\"ok\":true,\"engine\":\"llama\",\"llama_revision\":\"%s\","
      "\"prefix\":%d,\"eval_count\":%d,\"emitted_token_count\":0,"
      "\"capacity\":%d,\"window_steps\":%d,\"nsys_capture\":%s,"
      "\"prefill_ms\":%.9g,\"decode_only_ms\":%.9g,"
      "\"complete_request_ms\":%.9g,\"keep\":false}\n",
      kPrefix, options.workload, kRevision, options.prefix, options.tokens,
      options.ctx, window_steps, json_bool(capture), prefill_ms, decode_only_ms,
      complete_request_ms);

  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
