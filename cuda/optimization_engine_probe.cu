#include "ffn_decode_path.cuh"
#include "full_scheduler.h"
#include "q4k_decode_path.cuh"
#include "q8_decode_path.cuh"
#include "quant_mmv.h"
#include "test_tier.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "mixer.h"
#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT057_PROBE_RESULT=";
constexpr std::size_t kTinyRows = 17;
constexpr std::size_t kTinyColumns = 256;
constexpr std::size_t kSmokeWarmups = 1;
constexpr std::size_t kSmokeSamples = 1;
constexpr std::size_t kCorrectnessTokens = 2;
constexpr std::size_t kScreenPrefix = 2048;
constexpr std::size_t kScreenOutputTokens = 32;
constexpr std::size_t kScreenPrompt = 4096;
constexpr char kDefaultSelector[] = "ffn_only";

struct Options final {
  const char* model = nullptr;
  const char* workload = nullptr;
  std::size_t prompt = 0;
  std::size_t prefix = 0;
  std::size_t output_tokens = 0;
  int runs = 1;
  const char* selector = kDefaultSelector;
  bool graph = false;
  bool eager = false;
  bool skip_logits = false;
  bool modes_set = false;
};

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

bool parse_size(const char* text, std::size_t* value) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<std::size_t>(parsed);
  return true;
}

bool parse_modes(const char* text, Options* options) {
  options->graph = false;
  options->eager = false;
  options->modes_set = true;
  std::string raw(text);
  std::size_t start = 0;
  while (start <= raw.size()) {
    const std::size_t comma = raw.find(',', start);
    const std::string token =
        raw.substr(start, comma == std::string::npos ? std::string::npos
                                                     : comma - start);
    if (token == "graph") {
      options->graph = true;
    } else if (token == "eager") {
      options->eager = true;
    } else if (!token.empty()) {
      return false;
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return options->graph || options->eager;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [MODEL.gguf] [--workload tiny|tokens|prefill|decode] "
               "[--prompt N] [--prefix N] [--output-tokens N] [--runs N] "
               "[--selector NAME] [--modes graph,eager] [--skip-logits]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--prompt") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->prompt)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->prefix)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--output-tokens") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->output_tokens)) {
        return usage(argv[0]);
      }
    } else if (std::strcmp(arg, "--runs") == 0 && index + 1 < argc) {
      std::size_t runs = 0;
      if (!parse_size(argv[++index], &runs) || runs == 0 || runs > 8) {
        std::fprintf(stderr, "--runs must be 1..8\n");
        return 2;
      }
      options->runs = static_cast<int>(runs);
    } else if (std::strcmp(arg, "--selector") == 0 && index + 1 < argc) {
      options->selector = argv[++index];
    } else if (std::strcmp(arg, "--modes") == 0 && index + 1 < argc) {
      if (!parse_modes(argv[++index], options)) {
        std::fprintf(stderr, "--modes must be graph, eager, or graph,eager\n");
        return 2;
      }
    } else if (std::strcmp(arg, "--skip-logits") == 0) {
      options->skip_logits = true;
    } else if (arg[0] != '-' && options->model == nullptr) {
      options->model = arg;
    } else {
      return usage(argv[0]);
    }
  }
  return 0;
}

int apply_defaults(const qw38::cuda::TestTier tier, Options* options) {
  if (options->workload == nullptr) {
    if (tier == qw38::cuda::TestTier::kSmoke) {
      options->workload = "tiny";
    } else if (tier == qw38::cuda::TestTier::kCorrectness) {
      options->workload = "tokens";
    } else {
      options->workload = "decode";
    }
  }
  if (!options->modes_set) {
    if (std::strcmp(options->workload, "tiny") == 0) {
      options->graph = false;
      options->eager = false;
    } else if (std::strcmp(options->workload, "tokens") == 0) {
      options->graph = true;
      options->eager = true;
    } else {
      options->graph = true;
      options->eager = true;
    }
  }
  if (std::strcmp(options->workload, "tiny") == 0) {
    if (options->output_tokens == 0 && options->prompt == 0 &&
        options->prefix == 0) {
      return 0;
    }
  }
  if (std::strcmp(options->workload, "tokens") == 0 &&
      options->output_tokens == 0) {
    options->output_tokens = kCorrectnessTokens;
  }
  if (std::strcmp(options->workload, "decode") == 0) {
    if (options->prefix == 0) options->prefix = kScreenPrefix;
    if (options->output_tokens == 0) {
      options->output_tokens = kScreenOutputTokens;
    }
  }
  if (std::strcmp(options->workload, "prefill") == 0 && options->prompt == 0) {
    options->prompt = kScreenPrompt;
  }
  return 0;
}

int reject_over_bounds(qw38::cuda::TestTier tier, const Options& options) {
  const bool tiny = std::strcmp(options.workload, "tiny") == 0;
  const bool tokens = std::strcmp(options.workload, "tokens") == 0;
  const bool prefill = std::strcmp(options.workload, "prefill") == 0;
  const bool decode = std::strcmp(options.workload, "decode") == 0;
  if (!tiny && !tokens && !prefill && !decode) {
    std::fprintf(stderr, "unknown workload %s\n", options.workload);
    return 2;
  }
  if (std::strcmp(options.selector, kDefaultSelector) != 0) {
    std::fprintf(stderr, "selector %s is not an admitted combination\n",
                 options.selector);
    return 2;
  }
  if (options.skip_logits) {
    std::fprintf(stderr,
                 "--skip-logits is rejected for the standard short probe\n");
    return 2;
  }
  if (tier == qw38::cuda::TestTier::kSmoke) {
    if (!tiny || options.model != nullptr || options.prompt != 0 ||
        options.prefix != 0 || options.output_tokens != 0 ||
        options.runs != 1) {
      std::fprintf(stderr, "smoke rejects model-scale or oversized requests\n");
      return 2;
    }
    return 0;
  }
  if (tier == qw38::cuda::TestTier::kCorrectness) {
    if (!tokens || options.prompt != 0 || options.prefix != 0 ||
        options.output_tokens > kCorrectnessTokens || options.runs != 1 ||
        options.model == nullptr) {
      std::fprintf(stderr,
                   "correctness requires MODEL, two tokens, and no P/D\n");
      return 2;
    }
    return 0;
  }
  if (tier == qw38::cuda::TestTier::kScreen) {
    const bool decode_ok =
        decode && options.prefix == kScreenPrefix &&
        options.output_tokens == kScreenOutputTokens && options.prompt == 0;
    const bool prefill_ok = prefill && options.prompt == kScreenPrompt &&
                            options.prefix == 0 && options.output_tokens == 0;
    if (options.model == nullptr || options.runs != 1 ||
        !(decode_ok || prefill_ok)) {
      std::fprintf(stderr,
                   "screen allows one pair of P4096 or prefix 2048 + 32 "
                   "output tokens\n");
      return 2;
    }
    return 0;
  }
  std::fprintf(stderr, "acceptance/release oracles are separate binaries\n");
  return 2;
}

void fill_tiny_weights(std::vector<std::uint8_t>* weights) {
  constexpr std::size_t kBlockBytes = 144;
  constexpr std::size_t kBlockValues = 256;
  weights->assign(kTinyRows * (kTinyColumns / kBlockValues) * kBlockBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kBlockBytes) {
    (*weights)[offset] = 0x00;
    (*weights)[offset + 1] = 0x24;
    (*weights)[offset + 2] = 0x00;
    (*weights)[offset + 3] = 0x1C;
  }
}

void fill_tiny_activation(std::vector<__nv_bfloat16>* activation,
                          std::vector<qw38::cuda::Q8Block>* staged) {
  activation->resize(kTinyColumns);
  staged->resize(kTinyColumns / 32);
  for (std::size_t index = 0; index < kTinyColumns; ++index) {
    const float phase = static_cast<float>(index) * 0.071F;
    (*activation)[index] =
        __float2bfloat16_rn(std::sin(phase) * 3.0F);
  }
}

int run_tiny() {
  std::vector<std::uint8_t> weights;
  std::vector<__nv_bfloat16> activation;
  std::vector<qw38::cuda::Q8Block> staged;
  fill_tiny_weights(&weights);
  fill_tiny_activation(&activation, &staged);

  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_activation = nullptr;
  qw38::cuda::Q8Block* device_staged = nullptr;
  float* device_output = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_activation,
                       activation.size() * sizeof(activation[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_staged, staged.size() * sizeof(staged[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output, kTinyRows * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_activation, activation.data(),
                       activation.size() * sizeof(activation[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("cudaMemcpy H2D", error);

  for (std::size_t warmup = 0; warmup < kSmokeWarmups; ++warmup) {
    error = qw38::cuda::launch_quant_mmv(
        qw38::cuda::QuantKind::kQ4K, device_weights, kTinyRows, kTinyColumns,
        device_activation, device_staged, device_output, nullptr);
    if (error != cudaSuccess) return fail_cuda("tiny warmup", error);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  for (std::size_t sample = 0; sample < kSmokeSamples && error == cudaSuccess;
       ++sample) {
    error = qw38::cuda::launch_quant_mmv(
        qw38::cuda::QuantKind::kQ4K, device_weights, kTinyRows, kTinyColumns,
        device_activation, device_staged, device_output, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  float milliseconds = 0.0F;
  if (error == cudaSuccess) {
    error = cudaEventElapsedTime(&milliseconds, start, stop);
  }
  if (start != nullptr) cudaEventDestroy(start);
  if (stop != nullptr) cudaEventDestroy(stop);
  if (error != cudaSuccess) return fail_cuda("tiny measure", error);

  std::vector<float> actual(kTinyRows);
  error = cudaMemcpy(actual.data(), device_output, kTinyRows * sizeof(float),
                     cudaMemcpyDeviceToHost);
  cudaFree(device_weights);
  cudaFree(device_activation);
  cudaFree(device_staged);
  cudaFree(device_output);
  if (error != cudaSuccess) return fail_cuda("cudaMemcpy D2H", error);
  for (float value : actual) {
    if (!std::isfinite(value)) {
      std::fprintf(stderr, "tiny projection produced a nonfinite value\n");
      return 1;
    }
  }
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-057\","
              "\"workload\":\"tiny\",\"rows\":%zu,\"columns\":%zu,"
              "\"warmups\":%zu,\"samples\":%zu,\"wall_ms\":%.9g,"
              "\"model\":false,\"logits_copied\":false}\n",
              kPrefix, kTinyRows, kTinyColumns, kSmokeWarmups, kSmokeSamples,
              static_cast<double>(milliseconds));
  std::printf("status=passed\n");
  return 0;
}

int load_model(const char* path, qw38::cuda::ResidentModel* model) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (status.is_ok()) {
    status = model->upload(weights, mapping.data(), mapping.size());
  }
  if (!status.is_ok()) return fail_status(status);
  return 0;
}

qw38::Status run_tokens(const qw38::cuda::ResidentModel& model,
                        const std::size_t* tokens, std::size_t prefix,
                        std::size_t output_tokens,
                        qw38::cuda::SchedulerSession* session,
                        qw38::cuda::SchedulerWorkspace* workspace,
                        qw38::cuda::SchedulerGraphs* graphs, float* logits,
                        float* hidden, float* wall_ms) {
  qw38::Status status = session->reset();
  if (!status.is_ok()) return status;
  if (prefix > 0) {
    qw38::cuda::SyncResult result{};
    status = qw38::cuda::sync_tokens(
        model, tokens, prefix, session, workspace, logits,
        qw38::internal::kVocabularySize, hidden, qw38::internal::kResidualWidth,
        &result, nullptr, graphs);
    if (!status.is_ok()) return status;
  }
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; step < output_tokens; ++step) {
    float elapsed = 0.0F;
    status = qw38::cuda::execute_token(
        model, tokens[prefix + step], session, workspace, logits,
        qw38::internal::kVocabularySize, hidden, qw38::internal::kResidualWidth,
        &elapsed, nullptr, nullptr, qw38::cuda::PointwisePath::kFused, graphs,
        nullptr);
    if (!status.is_ok()) return status;
  }
  if (wall_ms != nullptr) {
    *wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - started)
                                      .count());
  }
  return status;
}

qw38::Status run_prefill(const qw38::cuda::ResidentModel& model,
                         const std::size_t* tokens, std::size_t count,
                         qw38::cuda::SchedulerSession* session,
                         qw38::cuda::SchedulerWorkspace* workspace,
                         qw38::cuda::SchedulerGraphs* graphs, float* logits,
                         float* hidden, float* wall_ms) {
  qw38::Status status = session->reset();
  if (!status.is_ok()) return status;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  if (error != cudaSuccess) {
    return {qw38::StatusCode::kInternal, cudaGetErrorString(error)};
  }
  qw38::cuda::SyncResult result{};
  status = qw38::cuda::sync_tokens(
      model, tokens, count, session, workspace, logits,
      qw38::internal::kVocabularySize, hidden, qw38::internal::kResidualWidth,
      &result, nullptr, graphs);
  if (status.is_ok()) {
    error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(wall_ms, start, stop);
    }
  }
  if (start != nullptr) cudaEventDestroy(start);
  if (stop != nullptr) cudaEventDestroy(stop);
  if (status.is_ok() && error != cudaSuccess) {
    return {qw38::StatusCode::kInternal, cudaGetErrorString(error)};
  }
  return status;
}

bool exact_logits(const std::vector<float>& left,
                  const std::vector<float>& right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0;
}

int run_engine(const Options& options, bool correctness) {
  qw38::cuda::ResidentModel model;
  const int loaded = load_model(options.model, &model);
  if (loaded != 0) return loaded;

  const std::size_t prefix = options.prefix;
  const std::size_t output_tokens = options.output_tokens;
  const std::size_t prompt = options.prompt;
  const std::size_t token_count =
      prompt > 0 ? prompt : prefix + output_tokens;
  const std::size_t capacity =
      prompt > 0 ? prompt : std::max(prefix + output_tokens, std::size_t{64});
  std::vector<std::size_t> tokens(token_count);
  fill_tokens(&tokens);

  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = workspace.create(capacity);
  qw38::cuda::SchedulerSession graph_session;
  qw38::cuda::SchedulerSession eager_session;
  if (status.is_ok()) status = graph_session.create(capacity);
  if (status.is_ok()) status = eager_session.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok() && options.graph) {
    status = graphs.create(model, &workspace);
  }
  if (!status.is_ok()) return fail_status(status);

  qw38::cuda::set_mmq_pipeline_path_override("off");
  const char* captured_path = graphs.execution_graph_path();
  const bool override_ignored =
      std::strcmp(captured_path, kDefaultSelector) == 0;
  qw38::cuda::set_mmq_pipeline_path_override(nullptr);

  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::vector<float> replay_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  std::array<float, qw38::internal::kResidualWidth> replay_hidden{};
  float graph_ms = 0.0F;
  float eager_ms = 0.0F;
  std::size_t graph_frontier = 0;
  std::size_t eager_frontier = 0;

  auto execute = [&](qw38::cuda::SchedulerSession* session,
                     qw38::cuda::SchedulerGraphs* graph_ptr, float* logits,
                     float* hidden, float* wall) {
    if (prompt > 0) {
      return run_prefill(model, tokens.data(), prompt, session, &workspace,
                         graph_ptr, logits, hidden, wall);
    }
    return run_tokens(model, tokens.data(), prefix, output_tokens, session,
                      &workspace, graph_ptr, logits, hidden, wall);
  };

  if (options.graph) {
    status = execute(&graph_session, &graphs, graph_logits.data(),
                     graph_hidden.data(), &graph_ms);
    if (!status.is_ok()) return fail_status(status);
    graph_frontier = graph_session.frontier();
    if (correctness) {
      status = execute(&graph_session, &graphs, replay_logits.data(),
                       replay_hidden.data(), nullptr);
      if (!status.is_ok()) return fail_status(status);
      if (!exact_logits(graph_logits, replay_logits) ||
          std::memcmp(graph_hidden.data(), replay_hidden.data(),
                      graph_hidden.size() * sizeof(float)) != 0 ||
          graph_session.frontier() != graph_frontier) {
        std::fprintf(stderr, "reset/replay logits or frontier diverged\n");
        return 1;
      }
    }
  }
  if (options.eager) {
    status = execute(&eager_session, nullptr, eager_logits.data(),
                     eager_hidden.data(), &eager_ms);
    if (!status.is_ok()) return fail_status(status);
    eager_frontier = eager_session.frontier();
  }
  if (options.graph && options.eager) {
    if (!exact_logits(graph_logits, eager_logits) ||
        graph_frontier != eager_frontier) {
      std::fprintf(stderr, "graph/eager logits or frontier diverged\n");
      return 1;
    }
    bool equal = false;
    status = graph_session.state_equals(eager_session, &equal);
    if (!status.is_ok()) return fail_status(status);
    if (!equal) {
      std::fprintf(stderr, "graph/eager session state diverged\n");
      return 1;
    }
  }

  const std::size_t timed_tokens = prompt > 0 ? prompt : output_tokens;
  const float control_tok_s =
      graph_ms > 0.0F ? static_cast<float>(timed_tokens) * 1000.0F / graph_ms
                      : 0.0F;
  const float candidate_tok_s =
      eager_ms > 0.0F ? static_cast<float>(timed_tokens) * 1000.0F / eager_ms
                      : 0.0F;
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-057\","
              "\"workload\":\"%s\",\"selector\":\"%s\",\"modes\":[",
              kPrefix, options.workload, options.selector);
  bool first = true;
  if (options.graph) {
    std::printf("\"graph\"");
    first = false;
  }
  if (options.eager) {
    if (!first) std::printf(",");
    std::printf("\"eager\"");
  }
  std::printf("],\"prompt\":%zu,\"prefix\":%zu,\"output_tokens\":%zu,"
              "\"runs\":%d,\"graph_wall_ms\":%.9g,\"eager_wall_ms\":%.9g,"
              "\"control_tok_s\":%.9g,\"candidate_tok_s\":%.9g,"
              "\"graph_frontier\":%zu,\"eager_frontier\":%zu,"
              "\"logits_copied\":true,\"p95_diagnostic\":%s,"
              "\"override_after_capture_ignored\":%s,"
              "\"captured_path\":\"%s\",\"q4_decode\":\"%s\","
              "\"q8_decode\":\"%s\",\"q8_rows_skinny\":%u,"
              "\"q8_layout_warps_skinny\":%u,\"ffn_decode\":\"%s\","
              "\"model_loaded_once\":true}\n",
              prompt, prefix, output_tokens, options.runs,
              static_cast<double>(graph_ms), static_cast<double>(eager_ms),
              static_cast<double>(control_tok_s),
              static_cast<double>(candidate_tok_s), graph_frontier,
              eager_frontier, correctness ? "false" : "true",
              override_ignored ? "true" : "false", captured_path,
              qw38::cuda::selected_q4_decode_path(),
              qw38::cuda::selected_q8_decode_path(),
              qw38::cuda::selected_q8_decode_rows_skinny(),
              qw38::cuda::selected_q8_decode_layout_warps_skinny(),
              qw38::cuda::selected_ffn_decode_path());
  std::printf("status=passed\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr,
                 "QW38_CUDA_TEST_TIER must be set to smoke, correctness, "
                 "screen, or acceptance\n");
    return 2;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  Options options;
  const int parsed = parse_args(argc, argv, &options);
  if (parsed != 0) return parsed;
  if (apply_defaults(tier, &options) != 0) return 2;
  const int bounded = reject_over_bounds(tier, options);
  if (bounded != 0) return bounded;
  if (std::strcmp(options.workload, "tiny") == 0) return run_tiny();
  return run_engine(options, tier == qw38::cuda::TestTier::kCorrectness);
}
