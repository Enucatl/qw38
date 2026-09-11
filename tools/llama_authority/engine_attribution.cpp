#include "ggml.h"
#include "llama.h"

#include <cuda_runtime.h>

#ifdef QW38_OPT060_ATTRIBUTION
#include "qw38_opt060_attribution.cuh"
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kVocabulary = 248320;
constexpr char kPrefix[] = "QW38_OPT060_LLAMA_ATTRIBUTION_RESULT=";
constexpr char kRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";

struct Options final {
  const char* model = nullptr;
  const char* workload = "smoke";
  const char* phase = "decode";
  const char* mode = "eager_diagnostic";
  const char* batch_policy = "matched";
  int prefix = 2048;
  int prompt = 4096;
  int output_tokens = 16;
  int warmups = 1;
  int samples = 1;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [MODEL.gguf] [--workload smoke|decode|prefill] "
               "[--phase decode|prefill] [--mode unperturbed|eager_diagnostic] "
               "[--batch-policy historical|matched] [--prefix N] [--prompt N] "
               "[--output-tokens N] [--warmups N] [--samples N]\n",
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
    } else if (std::strcmp(arg, "--phase") == 0 && index + 1 < argc) {
      options->phase = argv[++index];
    } else if (std::strcmp(arg, "--mode") == 0 && index + 1 < argc) {
      options->mode = argv[++index];
    } else if (std::strcmp(arg, "--batch-policy") == 0 && index + 1 < argc) {
      options->batch_policy = argv[++index];
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->prefix)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--prompt") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->prompt)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--output-tokens") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->output_tokens)) {
        return usage(argv[0]);
      }
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->warmups)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->samples)) return usage(argv[0]);
    } else if (arg[0] != '-' && options->model == nullptr) {
      options->model = arg;
    } else {
      return usage(argv[0]);
    }
  }
  return 0;
}

llama_token token_at(int index) {
  return static_cast<llama_token>((42 + index * 997) % kVocabulary);
}

int run_smoke() {
#ifdef QW38_OPT060_ATTRIBUTION
  setenv("QW38_OPT060_ATTRIBUTION", "1", 1);
  setenv("QW38_OPT060_EAGER", "1", 1);
  qw38_opt060::begin_run();
  qw38_opt060::set_positions("smoke", 0, 0);
  qw38_opt060::record_epoch(nullptr);
  qw38_opt060::begin_op(nullptr, 0);
  cudaEvent_t marker = nullptr;
  cudaEventCreate(&marker);
  cudaEventRecord(marker);
  qw38_opt060::end_graph();
  qw38_opt060::begin_op(nullptr, 0);
  cudaEventRecord(marker);
  cudaEventRecord(marker);
  qw38_opt060::end_graph();
  qw38_opt060::begin_op(nullptr, 0);
  cudaEventRecord(marker);
  qw38_opt060::end_graph();
  cudaEventDestroy(marker);
  qw38_opt060::resolve();
#endif
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-060\",\"engine\":\"llama\","
      "\"workload\":\"smoke\",\"revision\":\"%s\",\"fused_pair\":true,"
      "\"operations\":3,\"model\":false}\n",
      kPrefix, kRevision);
#ifdef QW38_OPT060_ATTRIBUTION
  qw38_opt060::dump_json(stdout);
#endif
  return 0;
}

void configure_batch(llama_context_params* params, const Options& options) {
  if (std::strcmp(options.batch_policy, "historical") == 0) {
    params->n_batch = 2048;
    params->n_ubatch = 512;
    return;
  }
  if (std::strcmp(options.phase, "prefill") == 0 ||
      std::strcmp(options.workload, "prefill") == 0) {
    params->n_batch = 4096;
    params->n_ubatch = 4096;
  } else {
    params->n_batch = 2048;
    params->n_ubatch = 1;
  }
}

int run_model(const Options& options) {
  if (options.model == nullptr) {
    std::fprintf(stderr, "model path required\n");
    return 2;
  }
  const bool diagnostic =
      std::strcmp(options.mode, "eager_diagnostic") == 0 ||
      std::strcmp(options.mode, "diagnostic") == 0;
  if (diagnostic) {
    setenv("QW38_OPT060_ATTRIBUTION", "1", 1);
    setenv("QW38_OPT060_EAGER", "1", 1);
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
  } else {
    unsetenv("QW38_OPT060_EAGER");
    unsetenv("GGML_CUDA_DISABLE_GRAPHS");
    if (std::strcmp(options.mode, "unperturbed") != 0) {
      std::fprintf(stderr, "mode must be unperturbed or eager_diagnostic\n");
      return 2;
    }
  }

  llama_backend_init();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 99;
  llama_model* model = llama_model_load_from_file(options.model, model_params);
  if (model == nullptr) {
    llama_backend_free();
    return 1;
  }
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = 4096;
  context_params.n_seq_max = 1;
  context_params.no_perf = true;
  configure_batch(&context_params, options);

  const bool prefill = std::strcmp(options.workload, "prefill") == 0 ||
                       std::strcmp(options.phase, "prefill") == 0;
  const int prompt = prefill ? options.prompt : options.prefix;
  const int decode_tokens = prefill ? 0 : options.output_tokens;
  std::vector<llama_token> tokens(static_cast<std::size_t>(prompt + decode_tokens));
  for (int index = 0; index < prompt + decode_tokens; ++index) {
    tokens[static_cast<std::size_t>(index)] = token_at(index);
  }

  double uninstrumented_ms = 0.0;
  double instrumented_ms = 0.0;
#ifdef QW38_OPT060_ATTRIBUTION
  qw38_opt060::begin_run();
  qw38_opt060::set_positions(prefill ? "prefill" : "decode", 0, 0);
#endif

  llama_context* ctx = llama_init_from_model(model, context_params);
  if (ctx == nullptr) {
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  for (int warm = 0; warm < options.warmups; ++warm) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_get_one(tokens.data(), prompt);
    if (llama_decode(ctx, batch) != 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
  }

  llama_memory_clear(llama_get_memory(ctx), true);
  if (!prefill) {
    llama_batch prefix_batch = llama_batch_get_one(tokens.data(), prompt);
    if (llama_decode(ctx, prefix_batch) != 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
  }
#ifdef QW38_OPT060_ATTRIBUTION
  if (diagnostic) qw38_opt060::record_epoch(nullptr);
#endif
  const auto started = std::chrono::steady_clock::now();
  if (prefill) {
    llama_batch prompt_batch = llama_batch_get_one(tokens.data(), prompt);
    if (llama_decode(ctx, prompt_batch) != 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
  }
  for (int step = 0; step < decode_tokens; ++step) {
#ifdef QW38_OPT060_ATTRIBUTION
    qw38_opt060::set_positions("decode", 0, prompt + step);
#endif
    llama_token token = tokens[static_cast<std::size_t>(prompt + step)];
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch) != 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    const float* logits = llama_get_logits_ith(ctx, -1);
    volatile float sink = logits != nullptr ? logits[0] : 0.0F;
    (void)sink;
  }
  cudaDeviceSynchronize();
  const double wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  if (diagnostic) {
    instrumented_ms = wall_ms;
  } else {
    uninstrumented_ms = wall_ms;
  }
#ifdef QW38_OPT060_ATTRIBUTION
  if (diagnostic) qw38_opt060::resolve();
#endif
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();

  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-060\",\"engine\":\"llama\","
      "\"workload\":\"%s\",\"mode\":\"%s\",\"batch_policy\":\"%s\","
      "\"revision\":\"%s\",\"graph_mode\":\"%s\",\"n_batch\":%u,\"n_ubatch\":%u,"
      "\"prompt\":%d,\"output_tokens\":%d,\"uninstrumented_wall_ms\":%.9g,"
      "\"instrumented_wall_ms\":%.9g,\"not_llama_bench\":true}\n",
      kPrefix, options.workload, options.mode, options.batch_policy, kRevision,
      diagnostic ? "eager_diagnostic" : "cuda_graph",
      static_cast<unsigned>(context_params.n_batch),
      static_cast<unsigned>(context_params.n_ubatch), prompt, decode_tokens,
      uninstrumented_ms, instrumented_ms);
#ifdef QW38_OPT060_ATTRIBUTION
  if (diagnostic) qw38_opt060::dump_json(stdout);
#endif
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  if (std::strcmp(options.workload, "smoke") == 0) return run_smoke();
  return run_model(options);
}
