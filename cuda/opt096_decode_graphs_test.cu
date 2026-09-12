#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "test_tier.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT096_DECODE_GRAPHS_RESULT=";
constexpr float kOverheadTriggerMs = 0.5F;

struct Options final {
  const char* workload = nullptr;
  const char* execution_graphs = "ffn_only";
  std::size_t prefix = 128;
  std::size_t warmups = 1;
  std::size_t samples = 3;
  std::size_t tokens = 1;
};

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

qw38::Status poll(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls == context->stop_after) {
    return {qw38::StatusCode::kCancelled, "opt096 graph cancellation"};
  }
  return qw38::Status::ok();
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] =
        (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

bool exact_buffers(const std::vector<float>& left,
                   const std::vector<float>& right,
                   const std::array<float, qw38::internal::kResidualWidth>& hidden_left,
                   const std::array<float, qw38::internal::kResidualWidth>& hidden_right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0 &&
         std::memcmp(hidden_left.data(), hidden_right.data(),
                     hidden_left.size() * sizeof(float)) == 0;
}

qw38::Status run_token(const qw38::cuda::ResidentModel& model,
                       std::size_t token, qw38::cuda::SchedulerSession* session,
                       qw38::cuda::SchedulerWorkspace* workspace,
                       float* logits, float* hidden, float* elapsed,
                       qw38::cuda::SchedulerGraphs* graphs,
                       const qw38::cuda::EvalControl* control,
                       qw38::cuda::RuntimeTimings* timings = nullptr,
                       qw38::cuda::DecodeAttribution* attribution = nullptr) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, control, timings,
      qw38::cuda::PointwisePath::kFused, graphs, attribution);
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|screen|benchmark|overhead] "
               "[--execution-graphs ffn_only|decode_segments8] "
               "[--prefix N] [--warmups N] [--samples N] [--tokens N] MODEL\n",
               argv0);
  return 2;
}

bool parse_size(const char* text, std::size_t* value) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<std::size_t>(parsed);
  return true;
}

int parse_args(int argc, char** argv, Options* options) {
  int model_index = -1;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--execution-graphs") == 0 && index + 1 < argc) {
      options->execution_graphs = argv[++index];
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->prefix)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->warmups)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->samples)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--tokens") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->tokens)) return usage(argv[0]);
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    } else {
      model_index = index;
    }
  }
  if (model_index < 0) return usage(argv[0]);
  if (!qw38::cuda::legal_execution_graph_path(options->execution_graphs)) {
    std::fprintf(stderr, "invalid --execution-graphs %s\n",
                 options->execution_graphs);
    return 2;
  }
  return model_index;
}

void print_dispatch(const qw38::cuda::SchedulerGraphs& graphs) {
  std::printf(
      "decode_graph_dispatch path=%s decode_graph_count=%zu "
      "decode_segment_graph_count=%zu launch_param_updates=%u\n",
      graphs.execution_graph_path(), graphs.decode_graph_count(),
      graphs.decode_segment_graph_count(), graphs.launch_param_update_count());
}

float measure_idle(const qw38::cuda::ResidentModel& model, std::size_t prefix) {
  const std::size_t capacity = std::max(prefix + 32, std::size_t{4096});
  std::vector<std::size_t> tokens(prefix + 1);
  fill_tokens(&tokens);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerSession session;
  qw38::Status status = workspace.create(capacity);
  if (status.is_ok()) status = session.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphFfnOnly);
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (status.is_ok()) {
    qw38::cuda::SyncResult sync{};
    status = qw38::cuda::sync_tokens(
        model, tokens.data(), prefix, &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &sync, nullptr, &graphs);
  }
  if (!status.is_ok()) return 0.0F;
  qw38::cuda::DecodeAttribution attribution;
  float elapsed = 0.0F;
  status = run_token(model, tokens[prefix], &session, &workspace, logits.data(),
                     hidden.data(), &elapsed, &graphs, nullptr, nullptr,
                     &attribution);
  if (!status.is_ok()) return 0.0F;
  return attribution.other_idle.milliseconds;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 1;
  }
  Options options;
  const int model_index = parse_args(argc, argv, &options);
  if (model_index < 0) return model_index == 0 ? 1 : model_index;
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  const char* workload = options.workload;
  if (workload == nullptr) {
    if (tier == qw38::cuda::TestTier::kSmoke) workload = "smoke";
    else if (tier == qw38::cuda::TestTier::kScreen) workload = "screen";
    else workload = "correctness";
  }

  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(argv[model_index], &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(argv[model_index]);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  qw38::cuda::ResidentModel model;
  if (status.is_ok()) {
    status = model.upload(weights, mapping.data(), mapping.size());
  }
  if (!status.is_ok()) return fail_status(status);

  if (std::strcmp(workload, "overhead") == 0) {
    const float d128 = measure_idle(model, 128);
    const float d2048 = measure_idle(model, 2048);
    std::printf("d128_idle_ms_per_token=%.9g d2048_idle_ms_per_token=%.9g\n",
                static_cast<double>(d128), static_cast<double>(d2048));
    std::printf("status=passed below_trigger=%s\n",
                json_bool(d128 < kOverheadTriggerMs ||
                          d2048 < kOverheadTriggerMs));
    return 0;
  }

  qw38::cuda::ExecutionGraphPathScope scope(options.execution_graphs);
  const std::size_t capacity =
      tier == qw38::cuda::TestTier::kSmoke ? 64 : 131072;
  qw38::cuda::SchedulerSession graph_session;
  qw38::cuda::SchedulerWorkspace graph_workspace;
  qw38::cuda::SchedulerSession eager_session;
  qw38::cuda::SchedulerWorkspace eager_workspace;
  status = graph_session.create(capacity);
  if (status.is_ok()) status = graph_workspace.create(capacity);
  if (status.is_ok()) status = eager_session.create(capacity);
  if (status.is_ok()) status = eager_workspace.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &graph_workspace);
  if (!status.is_ok()) return fail_status(status);

  print_dispatch(graphs);
  bool passed = true;
  const bool segments =
      std::strcmp(options.execution_graphs,
                  qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0;
  passed = passed &&
           ((segments && graphs.decode_segment_graph_count() == 8 &&
             graphs.decode_graph_count() == 0) ||
            (!segments && graphs.decode_graph_count() ==
                              qw38::internal::kModelLayerCount &&
             graphs.decode_segment_graph_count() == 0));

  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  float graph_ms = 0.0F;
  float eager_ms = 0.0F;
  status = run_token(model, 42, &graph_session, &graph_workspace,
                     graph_logits.data(), graph_hidden.data(), &graph_ms, &graphs,
                     nullptr, nullptr, nullptr);
  if (status.is_ok()) {
    status = run_token(model, 42, &eager_session, &eager_workspace,
                       eager_logits.data(), eager_hidden.data(), &eager_ms,
                       nullptr, nullptr, nullptr, nullptr);
  }
  if (!status.is_ok()) return fail_status(status);
  passed = passed && exact_buffers(graph_logits, eager_logits, graph_hidden,
                                   eager_hidden);

  PollContext cancel_ctx{0, 8};
  const qw38::cuda::EvalControl cancel_control{poll, &cancel_ctx};
  qw38::cuda::SchedulerSession cancel_session;
  status = cancel_session.create(capacity);
  if (status.is_ok()) {
    const qw38::Status cancelled = run_token(
        model, 11, &cancel_session, &graph_workspace, graph_logits.data(),
        graph_hidden.data(), &graph_ms, &graphs, &cancel_control, nullptr,
        nullptr);
    passed = passed && cancelled.code() == qw38::StatusCode::kCancelled &&
              cancel_session.frontier() == 0 && cancel_ctx.calls == 8;
  }

  if (std::strcmp(workload, "benchmark") == 0) {
    for (std::size_t sample = 0; sample < options.warmups + options.samples;
         ++sample) {
      std::vector<std::size_t> tokens(options.prefix + options.tokens);
      fill_tokens(&tokens);
      qw38::cuda::SchedulerSession session;
      status = session.create(capacity);
      if (status.is_ok()) {
        qw38::cuda::SyncResult sync{};
        status = qw38::cuda::sync_tokens(
            model, tokens.data(), options.prefix, &session, &graph_workspace,
            graph_logits.data(), graph_logits.size(), graph_hidden.data(),
            graph_hidden.size(), &sync, nullptr, &graphs);
      }
      if (!status.is_ok()) return fail_status(status);
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      cudaError_t error = cudaEventCreate(&start);
      if (error == cudaSuccess) error = cudaEventCreate(&stop);
      if (error == cudaSuccess) error = cudaEventRecord(start);
      float elapsed = 0.0F;
      for (std::size_t index = 0;
           error == cudaSuccess && index < options.tokens; ++index) {
        status = run_token(model, tokens[options.prefix + index], &session,
                           &graph_workspace, graph_logits.data(),
                           graph_hidden.data(), &elapsed, &graphs, nullptr,
                           nullptr, nullptr);
        if (!status.is_ok()) return fail_status(status);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      if (error == cudaSuccess) error = cudaEventElapsedTime(&elapsed, start, stop);
      if (start != nullptr) cudaEventDestroy(start);
      if (stop != nullptr) cudaEventDestroy(stop);
      if (error != cudaSuccess) return 1;
      if (sample >= options.warmups) {
        std::printf(
            "{\"cache_mode\":\"rotating\",\"sample_index\":%zu,"
            "\"enclosing_ms\":%.9g}\n",
            sample - options.warmups, static_cast<double>(elapsed));
      }
    }
  }

  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);
  std::printf("status=%s pass=%s measurement_utc=%s\n",
              passed ? "passed" : "failed", json_bool(passed), utc);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-096\",\"status\":\"%s\","
      "\"execution_graphs\":\"%s\",\"decode_graph_count\":%zu,"
      "\"decode_segment_graph_count\":%zu,\"launch_param_updates\":%u}\n",
      kPrefix, passed ? "passed" : "failed", options.execution_graphs,
      graphs.decode_graph_count(), graphs.decode_segment_graph_count(),
      graphs.launch_param_update_count());
  return passed ? 0 : 1;
}
