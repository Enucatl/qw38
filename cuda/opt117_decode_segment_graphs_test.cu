#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "opt110_engine_hook.cuh"
#include "test_tier.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT117_DECODE_SEGMENT_GRAPHS_RESULT=";
constexpr char kCounts[] = "QW38_OPT117_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;

struct Options final {
  const char* workload = "capture-repro";
  const char* execution_graphs = "decode_segments8";
  std::size_t prefix = 128;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
  std::size_t capacity = 0;
};

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
};

struct ArmResult final {
  float wall_ms = 0.0F;
  float tok_s = 0.0F;
  float p50_ms = 0.0F;
  float p95_ms = 0.0F;
  std::size_t graph_bytes = 0;
  std::uint32_t launch_param_updates = 0;
  std::size_t decode_segment_graphs = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s [--workload capture-repro|positions|pointer-swaps|"
      "cancellation|same-math|graph-ab|state-memory] "
      "[--execution-graphs ffn_only|decode_segments8] "
      "[--prefix N] [--warmups N] [--samples N] [--tokens N] [--capacity N] "
      "MODEL\n",
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
    } else if (std::strcmp(arg, "--capacity") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->capacity)) return usage(argv[0]);
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

std::size_t session_capacity(std::size_t prefix, std::size_t outputs,
                             std::size_t requested) {
  if (requested != 0) return requested;
  return std::max(prefix + outputs + 32, kMinCapacity);
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
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

qw38::Status load_model(const char* path, qw38::internal::MappedFile* mapping,
                        qw38::cuda::ResidentModel* model) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  if (status.is_ok()) status = mapping->open(path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, *mapping, &weights);
  }
  if (status.is_ok()) {
    status = model->upload(weights, mapping->data(), mapping->size());
  }
  return status;
}

qw38::Status run_token(const qw38::cuda::ResidentModel& model, std::size_t token,
                       qw38::cuda::SchedulerSession* session,
                       qw38::cuda::SchedulerWorkspace* workspace, float* logits,
                       float* hidden, float* elapsed,
                       qw38::cuda::SchedulerGraphs* graphs,
                       const qw38::cuda::EvalControl* control = nullptr) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, control, nullptr,
      qw38::cuda::PointwisePath::kFused, graphs, nullptr);
}

qw38::Status json_escape(const std::string& input, std::string* output) {
  output->clear();
  output->reserve(input.size());
  for (char ch : input) {
    if (ch == '"' || ch == '\\') {
      output->push_back('\\');
      output->push_back(ch);
    } else if (ch == '\n') {
      output->append("\\n");
    } else {
      output->push_back(ch);
    }
  }
  return qw38::Status::ok();
}

qw38::Status create_graphs(const qw38::cuda::ResidentModel& model,
                           qw38::cuda::SchedulerSession* session,
                           qw38::cuda::SchedulerWorkspace* workspace,
                           qw38::cuda::SchedulerGraphs* graphs,
                           const char* path) {
  qw38::cuda::ExecutionGraphPathScope scope(path);
  if (std::strcmp(path, qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0) {
    return graphs->create(model, workspace, session);
  }
  return graphs->create(model, workspace);
}

void print_counts(std::size_t warmups, std::size_t samples,
                  std::size_t candidates, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-117\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

int run_capture_repro(const qw38::cuda::ResidentModel& model,
                      const Options& options) {
  const std::size_t capacity =
      session_capacity(options.prefix, 1, options.capacity);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (!status.is_ok()) return fail_status(status);

  cudaStream_t stream = nullptr;
  cudaError_t stream_error =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (stream_error != cudaSuccess) {
    std::fprintf(stderr, "cudaStreamCreate failed: %s\n",
                 cudaGetErrorName(stream_error));
    return 1;
  }
  cudaGraph_t graph = nullptr;
  cudaError_t enqueue_error = cudaSuccess;
  cudaError_t end_error = cudaSuccess;
  const cudaError_t capture_error = qw38::cuda::capture_decode_segment_graph(
      model, 0, &session, &workspace, 1, stream, &graph, &enqueue_error,
      &end_error);
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);

  qw38::cuda::SchedulerGraphs graphs;
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  const auto create_started = std::chrono::steady_clock::now();
  status = graphs.create(model, &workspace, &session);
  const float create_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - create_started)
          .count());
  std::string message;
  json_escape(status.message(), &message);
  const bool ok = status.is_ok() &&
                  graphs.decode_segment_graph_count() == 8 &&
                  graphs.decode_graph_count() == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-117\","
      "\"workload\":\"capture-repro\",\"ok\":%s,"
      "\"segment_enqueue_error\":\"%s\",\"segment_end_capture_error\":\"%s\","
      "\"segment_capture_error\":\"%s\",\"create_ok\":%s,"
      "\"create_message\":\"%s\",\"create_ms\":%.9g,"
      "\"decode_segment_graph_count\":%zu,\"decode_graph_count\":%zu,"
      "\"decode_node_count\":%zu,\"graph_bytes\":%zu,"
      "\"opt110_hooks_ready\":%s,"
      "\"root_cause\":\"session_committed_vs_workspace_candidate\","
      "\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false}\n",
      kPrefix, json_bool(ok),
      enqueue_error == cudaSuccess ? "ok" : cudaGetErrorName(enqueue_error),
      end_error == cudaSuccess ? "ok" : cudaGetErrorName(end_error),
      capture_error == cudaSuccess ? "ok" : cudaGetErrorName(capture_error),
      json_bool(status.is_ok()), message.c_str(),
      static_cast<double>(create_ms), graphs.decode_segment_graph_count(),
      graphs.decode_graph_count(), graphs.decode_node_count(),
      graphs.allocated_bytes(),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
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

qw38::Status prefill(const qw38::cuda::ResidentModel& model,
                     const std::vector<std::size_t>& tokens, std::size_t prefix,
                     qw38::cuda::SchedulerSession* session,
                     qw38::cuda::SchedulerWorkspace* workspace,
                     qw38::cuda::SchedulerGraphs* graphs, float* logits,
                     float* hidden) {
  qw38::cuda::SyncResult sync{};
  return qw38::cuda::sync_tokens(model, tokens.data(), prefix, session,
                                 workspace, logits, qw38::internal::kVocabularySize,
                                 hidden, qw38::internal::kResidualWidth, &sync,
                                 nullptr, graphs);
}

int run_positions(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t positions[] = {0, 127, 1023, 1024, 2047, 131071};
  const std::size_t npos = sizeof(positions) / sizeof(positions[0]);
  bool all_ok = true;
  std::string body =
      "{\"schema_version\":1,\"task\":\"OPT-117\","
      "\"workload\":\"positions\",\"positions\":[";
  for (std::size_t index = 0; index < npos; ++index) {
    const std::size_t frontier = positions[index];
    const bool long_ctx = frontier >= 65536;
    const std::size_t prefix = long_ctx ? 0 : frontier;
    const std::size_t capacity =
        long_ctx ? 131072 : session_capacity(prefix + 1, 1, options.capacity);
    qw38::cuda::SchedulerGraphs ffn_graphs;
    qw38::cuda::SchedulerGraphs segment_graphs;
    qw38::cuda::SchedulerSession eager_session;
    qw38::cuda::SchedulerWorkspace eager_workspace;
    qw38::cuda::SchedulerSession graph_session;
    qw38::cuda::SchedulerWorkspace graph_workspace;
    qw38::Status status = graph_session.create(capacity);
    if (status.is_ok()) status = graph_workspace.create(capacity);
    bool ok = status.is_ok();
    bool exact = false;
    std::uint32_t updates = 0;
    std::string message;
    if (long_ctx) {
      if (ok) {
        status = create_graphs(model, &graph_session, &graph_workspace,
                               &segment_graphs,
                               qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      }
      ok = status.is_ok();
      if (ok) {
        const qw38::Status updated = segment_graphs.update_launch_params(
            0, static_cast<std::uint32_t>(frontier),
            static_cast<std::uint32_t>(frontier));
        ok = updated.is_ok();
        json_escape(updated.message(), &message);
        updates = segment_graphs.launch_param_update_count();
        exact = ok;
      } else {
        json_escape(status.message(), &message);
      }
    } else {
      if (ok) status = eager_session.create(capacity);
      if (status.is_ok()) status = eager_workspace.create(capacity);
      if (status.is_ok()) {
        status = create_graphs(model, &eager_session, &eager_workspace,
                               &ffn_graphs, qw38::cuda::kLegalExecutionGraphFfnOnly);
      }
      if (status.is_ok()) {
        status = create_graphs(model, &graph_session, &graph_workspace,
                               &segment_graphs,
                               qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      }
      std::vector<std::size_t> tokens(prefix + 1);
      fill_tokens(&tokens);
      std::vector<float> eager_logits(qw38::internal::kVocabularySize);
      std::vector<float> graph_logits(qw38::internal::kVocabularySize);
      std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
      std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
      if (status.is_ok() && prefix > 0) {
        status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                         &ffn_graphs, eager_logits.data(), eager_hidden.data());
        if (status.is_ok()) {
          status = prefill(model, tokens, prefix, &graph_session,
                           &graph_workspace, &segment_graphs, graph_logits.data(),
                           graph_hidden.data());
        }
      }
      float eager_ms = 0.0F;
      float graph_ms = 0.0F;
      if (status.is_ok()) {
        status = run_token(model, tokens[prefix], &eager_session, &eager_workspace,
                           eager_logits.data(), eager_hidden.data(), &eager_ms,
                           &ffn_graphs);
      }
      if (status.is_ok()) {
        qw38::cuda::ExecutionGraphPathScope scope(
            qw38::cuda::kLegalExecutionGraphDecodeSegments8);
        status = run_token(model, tokens[prefix], &graph_session, &graph_workspace,
                           graph_logits.data(), graph_hidden.data(), &graph_ms,
                           &segment_graphs);
      }
      ok = status.is_ok();
      json_escape(status.message(), &message);
      exact = ok && exact_buffers(eager_logits, graph_logits, eager_hidden,
                                  graph_hidden);
      updates = segment_graphs.launch_param_update_count();
    }
    all_ok = all_ok && ok && exact;
    char row[512];
    std::snprintf(
        row, sizeof(row),
        "%s{\"frontier\":%zu,\"ok\":%s,\"exact\":%s,\"launch_param_updates\":%u,"
        "\"decode_segment_graph_count\":%zu,\"message\":\"%s\"}",
        index == 0 ? "" : ",", frontier, json_bool(ok), json_bool(exact), updates,
        segment_graphs.decode_segment_graph_count(), message.c_str());
    body.append(row);
  }
  char tail[192];
  std::snprintf(tail, sizeof(tail),
                "],\"ok\":%s,\"observed_warmups\":0,\"observed_samples\":1,"
                "\"observed_candidates\":2,\"keep\":false}\n",
                json_bool(all_ok));
  body.append(tail);
  std::printf("%s%s", kPrefix, body.c_str());
  print_counts(0, 1, 2, false);
  return all_ok ? 0 : 1;
}

int run_pointer_swaps(const qw38::cuda::ResidentModel& model,
                      const Options& options) {
  const std::size_t prefix = options.prefix;
  const std::size_t capacity = session_capacity(prefix, 4, options.capacity);
  std::vector<std::size_t> tokens(prefix + 4);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession eager_session;
  qw38::cuda::SchedulerWorkspace eager_workspace;
  qw38::cuda::SchedulerSession graph_session;
  qw38::cuda::SchedulerWorkspace graph_workspace;
  qw38::cuda::SchedulerSession other_session;
  qw38::Status status = eager_session.create(capacity);
  if (status.is_ok()) status = eager_workspace.create(capacity);
  if (status.is_ok()) status = graph_session.create(capacity);
  if (status.is_ok()) status = graph_workspace.create(capacity);
  if (status.is_ok()) status = other_session.create(capacity);
  qw38::cuda::SchedulerGraphs ffn_graphs;
  qw38::cuda::SchedulerGraphs segment_graphs;
  if (status.is_ok()) {
    status = create_graphs(model, &eager_session, &eager_workspace, &ffn_graphs,
                           qw38::cuda::kLegalExecutionGraphFfnOnly);
  }
  if (status.is_ok()) {
    status = create_graphs(model, &graph_session, &graph_workspace,
                           &segment_graphs,
                           qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  }
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                     &ffn_graphs, eager_logits.data(), eager_hidden.data());
    if (status.is_ok()) {
      status = prefill(model, tokens, prefix, &graph_session, &graph_workspace,
                       &segment_graphs, graph_logits.data(), graph_hidden.data());
    }
  }
  bool exact = true;
  for (std::size_t step = 0; status.is_ok() && step < 2; ++step) {
    float eager_ms = 0.0F;
    float graph_ms = 0.0F;
    status = run_token(model, tokens[prefix + step], &eager_session,
                       &eager_workspace, eager_logits.data(), eager_hidden.data(),
                       &eager_ms, &ffn_graphs);
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix + step], &graph_session,
                         &graph_workspace, graph_logits.data(),
                         graph_hidden.data(), &graph_ms, &segment_graphs);
    }
    exact = exact && status.is_ok() &&
            exact_buffers(eager_logits, graph_logits, eager_hidden, graph_hidden);
  }
  bool invalidated = false;
  if (status.is_ok()) {
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    float graph_ms = 0.0F;
    const qw38::Status mismatch = run_token(
        model, tokens[prefix + 2], &other_session, &graph_workspace,
        graph_logits.data(), graph_hidden.data(), &graph_ms, &segment_graphs);
    invalidated = !mismatch.is_ok();
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool ok = status.is_ok() && exact && invalidated;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-117\","
      "\"workload\":\"pointer-swaps\",\"ok\":%s,\"exact\":%s,"
      "\"session_replace_invalidated\":%s,\"launch_param_updates\":%u,"
      "\"message\":\"%s\",\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(exact), json_bool(invalidated),
      segment_graphs.launch_param_update_count(), message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

qw38::Status poll_stop(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls >= context->stop_after) {
    return {qw38::StatusCode::kCancelled, "opt117 graph cancellation"};
  }
  return qw38::Status::ok();
}

int run_cancellation(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  const std::size_t prefix = options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  bool all_ok = true;
  std::string body =
      "{\"schema_version\":1,\"task\":\"OPT-117\","
      "\"workload\":\"cancellation\",\"cases\":[";
  const char* names[] = {"segment", "commit"};
  const std::size_t stops[] = {1, 8};
  for (int case_index = 0; case_index < 2; ++case_index) {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::Status status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    qw38::cuda::SchedulerGraphs graphs;
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    const std::size_t frontier_before = session.frontier();
    PollContext poll_context{};
    poll_context.stop_after = stops[case_index];
    qw38::cuda::EvalControl control{};
    control.poll = poll_stop;
    control.context = &poll_context;
    float elapsed = 0.0F;
    qw38::Status token_status = qw38::Status::ok();
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      token_status =
          run_token(model, tokens[prefix], &session, &workspace, logits.data(),
                    hidden.data(), &elapsed, &graphs, &control);
    }
    const bool cancelled = token_status.code() == qw38::StatusCode::kCancelled;
    const bool preserved = session.frontier() == frontier_before;
    const bool ok = status.is_ok() && cancelled && preserved &&
                    poll_context.calls >= 1;
    all_ok = all_ok && ok;
    char row[320];
    std::snprintf(
        row, sizeof(row),
        "%s{\"name\":\"%s\",\"ok\":%s,\"cancelled\":%s,\"frontier_preserved\":%s,"
        "\"poll_calls\":%zu,\"stop_after\":%zu}",
        case_index == 0 ? "" : ",", names[case_index], json_bool(ok),
        json_bool(cancelled), json_bool(preserved), poll_context.calls,
        stops[case_index]);
    body.append(row);
  }
  char tail[192];
  std::snprintf(tail, sizeof(tail),
                "],\"ok\":%s,\"observed_warmups\":0,\"observed_samples\":1,"
                "\"observed_candidates\":1,\"keep\":false}\n",
                json_bool(all_ok));
  body.append(tail);
  std::printf("%s%s", kPrefix, body.c_str());
  print_counts(0, 1, 1, false);
  return all_ok ? 0 : 1;
}

int run_same_math(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t prefix = options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 8 : options.tokens;
  const std::size_t capacity =
      session_capacity(prefix, outputs, options.capacity);
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession eager_session;
  qw38::cuda::SchedulerWorkspace eager_workspace;
  qw38::cuda::SchedulerSession graph_session;
  qw38::cuda::SchedulerWorkspace graph_workspace;
  qw38::Status status = eager_session.create(capacity);
  if (status.is_ok()) status = eager_workspace.create(capacity);
  if (status.is_ok()) status = graph_session.create(capacity);
  if (status.is_ok()) status = graph_workspace.create(capacity);
  qw38::cuda::SchedulerGraphs ffn_graphs;
  qw38::cuda::SchedulerGraphs segment_graphs;
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                     nullptr, eager_logits.data(), eager_hidden.data());
    if (status.is_ok()) {
      status = prefill(model, tokens, prefix, &graph_session, &graph_workspace,
                       nullptr, graph_logits.data(), graph_hidden.data());
    }
  }
  if (status.is_ok()) {
    status = create_graphs(model, &eager_session, &eager_workspace, &ffn_graphs,
                           qw38::cuda::kLegalExecutionGraphFfnOnly);
  }
  if (status.is_ok()) {
    status = create_graphs(model, &graph_session, &graph_workspace,
                           &segment_graphs,
                           qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  }
  const bool prefill_exact =
      status.is_ok() &&
      exact_buffers(eager_logits, graph_logits, eager_hidden, graph_hidden);
  bool exact = prefill_exact;
  std::size_t matched = 0;
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    float eager_ms = 0.0F;
    float graph_ms = 0.0F;
    status = run_token(model, tokens[prefix + step], &eager_session,
                       &eager_workspace, eager_logits.data(), eager_hidden.data(),
                       &eager_ms, nullptr);
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix + step], &graph_session,
                         &graph_workspace, graph_logits.data(),
                         graph_hidden.data(), &graph_ms, &segment_graphs);
    }
    const bool step_exact =
        status.is_ok() && exact_buffers(eager_logits, graph_logits, eager_hidden,
                                        graph_hidden);
    exact = exact && step_exact;
    if (step_exact) ++matched;
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool ok = status.is_ok() && exact &&
                  segment_graphs.decode_segment_graph_count() == 8;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-117\",\"workload\":\"same-math\","
      "\"ok\":%s,\"exact\":%s,\"prefill_exact\":%s,\"matched_tokens\":%zu,"
      "\"tokens\":%zu,"
      "\"prefix\":%zu,\"launch_param_updates\":%u,\"message\":\"%s\","
      "\"observed_warmups\":0,\"observed_samples\":1,\"observed_candidates\":2,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(exact), json_bool(prefill_exact), matched,
      outputs, prefix,
      segment_graphs.launch_param_update_count(), message.c_str());
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

qw38::Status run_prefill_arm(const qw38::cuda::ResidentModel& model,
                             const std::vector<std::size_t>& tokens,
                             std::size_t prompt, const char* graph_path,
                             ArmResult* out) {
  const std::size_t capacity = session_capacity(prompt, 0, 0);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) {
    status = create_graphs(model, &session, &workspace, &graphs, graph_path);
  }
  if (!status.is_ok()) return status;
  out->graph_bytes = graphs.allocated_bytes();
  out->decode_segment_graphs = graphs.decode_segment_graph_count();
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  const auto started = std::chrono::steady_clock::now();
  status = prefill(model, tokens, prompt, &session, &workspace, &graphs,
                   logits.data(), hidden.data());
  out->wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count());
  out->tok_s =
      out->wall_ms > 0.0F ? static_cast<float>(prompt) * 1000.0F / out->wall_ms
                          : 0.0F;
  out->p50_ms = out->wall_ms;
  out->p95_ms = out->wall_ms;
  out->launch_param_updates = graphs.launch_param_update_count();
  return status;
}

qw38::Status run_decode_arm(const qw38::cuda::ResidentModel& model,
                            const std::vector<std::size_t>& tokens,
                            std::size_t prefix, std::size_t outputs,
                            const char* graph_path, bool use_graphs,
                            ArmResult* out) {
  const std::size_t capacity = session_capacity(prefix, outputs, 0);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) {
    status = create_graphs(model, &session, &workspace, &graphs, graph_path);
  }
  if (!status.is_ok()) return status;
  out->graph_bytes = graphs.allocated_bytes();
  out->decode_segment_graphs = graphs.decode_segment_graph_count();
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                   logits.data(), hidden.data());
  if (!status.is_ok()) return status;
  qw38::cuda::SchedulerGraphs* launched =
      use_graphs ? &graphs : nullptr;
  std::vector<float> latencies;
  latencies.reserve(outputs);
  const auto started = std::chrono::steady_clock::now();
  qw38::cuda::ExecutionGraphPathScope scope(graph_path);
  for (std::size_t step = 0; step < outputs; ++step) {
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, launched);
    if (!status.is_ok()) return status;
    latencies.push_back(static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - token_started)
            .count()));
  }
  out->wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count());
  out->tok_s =
      out->wall_ms > 0.0F
          ? static_cast<float>(outputs) * 1000.0F / out->wall_ms
          : 0.0F;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  out->launch_param_updates = graphs.launch_param_update_count();
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"wall_ms\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,"
      "\"p95_ms\":%.9g,\"graph_bytes\":%zu,\"launch_param_updates\":%u,"
      "\"decode_segment_graphs\":%zu}%s",
      name, static_cast<double>(arm.wall_ms), static_cast<double>(arm.tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms),
      arm.graph_bytes, arm.launch_param_updates, arm.decode_segment_graphs,
      last ? "" : ",");
}

int run_graph_ab(const qw38::cuda::ResidentModel& model,
                 const Options& options) {
  const bool prefill_only = options.prefix == 4096;
  const std::size_t prompt = prefill_only ? 4096 : options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  const char* parent = qw38::cuda::kLegalExecutionGraphFfnOnly;
  const char* candidate = options.execution_graphs;
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    if (prefill_only) {
      status = run_prefill_arm(model, tokens, prompt, parent, &discarded);
      if (status.is_ok()) {
        status = run_prefill_arm(model, tokens, prompt, candidate, &discarded);
      }
    } else {
      status = run_decode_arm(model, tokens, prompt, outputs, parent, true,
                              &discarded);
      if (status.is_ok()) {
        status = run_decode_arm(model, tokens, prompt, outputs, candidate, true,
                                &discarded);
      }
    }
    if (!status.is_ok()) return fail_status(status);
    std::printf("quartz_warmup=%zu tok_s=%.9g wall_ms=%.9g independently_restored=true\n",
                warmup, static_cast<double>(discarded.tok_s),
                static_cast<double>(discarded.wall_ms));
  }
  std::vector<ArmResult> arm_a(options.samples);
  std::vector<ArmResult> arm_b(options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    const bool ba = (pair % 2) == 1;
    ArmResult first;
    ArmResult second;
    if (prefill_only) {
      status = run_prefill_arm(model, tokens, prompt, ba ? candidate : parent,
                               &first);
      if (status.is_ok()) {
        status = run_prefill_arm(model, tokens, prompt, ba ? parent : candidate,
                                 &second);
      }
    } else {
      status = run_decode_arm(model, tokens, prompt, outputs,
                              ba ? candidate : parent, true, &first);
      if (status.is_ok()) {
        status = run_decode_arm(model, tokens, prompt, outputs,
                                ba ? parent : candidate, true, &second);
      }
    }
    if (!status.is_ok()) return fail_status(status);
    arm_a[pair] = ba ? second : first;
    arm_b[pair] = ba ? first : second;
    std::printf(
        "quartz_pair sample_index=%zu order=%s A_tok_s=%.9g B_tok_s=%.9g "
        "independently_restored=true same_binary=true\n",
        pair, ba ? "BA" : "AB", static_cast<double>(arm_a[pair].tok_s),
        static_cast<double>(arm_b[pair].tok_s));
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-117\",\"workload\":\"graph-ab\","
      "\"prefix\":%zu,\"prefill\":%s,\"warmups\":%zu,\"samples\":%zu,"
      "\"tokens\":%zu,\"parent\":\"%s\",\"candidate\":\"%s\","
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"acceptance\",\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true,\"pairs\":[",
      kPrefix, prompt, json_bool(prefill_only), options.warmups, options.samples,
      outputs, parent, candidate, options.warmups, options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    if (pair != 0) std::printf(",");
    std::printf("{\"sample_index\":%zu,\"order\":\"%s\",", pair,
                (pair % 2) == 1 ? "BA" : "AB");
    print_arm("A", arm_a[pair], false);
    print_arm("B", arm_b[pair], true);
    std::printf("}");
  }
  std::printf("]}\n");
  print_counts(options.warmups, options.samples, 2, false);
  return 0;
}

int run_state_memory(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  const std::size_t prefix = options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession eager_session;
  qw38::cuda::SchedulerWorkspace eager_workspace;
  qw38::cuda::SchedulerSession graph_session;
  qw38::cuda::SchedulerWorkspace graph_workspace;
  qw38::Status status = eager_session.create(capacity);
  if (status.is_ok()) status = eager_workspace.create(capacity);
  if (status.is_ok()) status = graph_session.create(capacity);
  if (status.is_ok()) status = graph_workspace.create(capacity);
  qw38::cuda::SchedulerGraphs ffn_graphs;
  qw38::cuda::SchedulerGraphs segment_graphs;
  if (status.is_ok()) {
    status = create_graphs(model, &eager_session, &eager_workspace, &ffn_graphs,
                           qw38::cuda::kLegalExecutionGraphFfnOnly);
  }
  if (status.is_ok()) {
    status = create_graphs(model, &graph_session, &graph_workspace,
                           &segment_graphs,
                           qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  }
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                     &ffn_graphs, eager_logits.data(), eager_hidden.data());
    if (status.is_ok()) {
      status = prefill(model, tokens, prefix, &graph_session, &graph_workspace,
                       &segment_graphs, graph_logits.data(), graph_hidden.data());
    }
  }
  float eager_ms = 0.0F;
  float graph_ms = 0.0F;
  if (status.is_ok()) {
    status = run_token(model, tokens[prefix], &eager_session, &eager_workspace,
                       eager_logits.data(), eager_hidden.data(), &eager_ms,
                       &ffn_graphs);
  }
  if (status.is_ok()) {
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    status = run_token(model, tokens[prefix], &graph_session, &graph_workspace,
                       graph_logits.data(), graph_hidden.data(), &graph_ms,
                       &segment_graphs);
  }
  bool equal = false;
  if (status.is_ok()) {
    status = graph_session.state_equals(eager_session, &equal);
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool ok = status.is_ok() && equal &&
                  exact_buffers(eager_logits, graph_logits, eager_hidden,
                                graph_hidden);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-117\","
      "\"workload\":\"state-memory\",\"ok\":%s,\"state_equals\":%s,"
      "\"graph_bytes\":%zu,\"message\":\"%s\",\"observed_warmups\":0,"
      "\"observed_samples\":1,\"observed_candidates\":1,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(equal), segment_graphs.allocated_bytes(),
      message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  const int model_index = parse_args(argc, argv, &options);
  if (model_index < 0) return model_index == -1 ? 1 : model_index;
  qw38::internal::MappedFile mapping;
  qw38::cuda::ResidentModel model;
  const qw38::Status status = load_model(argv[model_index], &mapping, &model);
  if (!status.is_ok()) return fail_status(status);
  if (std::strcmp(options.workload, "capture-repro") == 0) {
    return run_capture_repro(model, options);
  }
  if (std::strcmp(options.workload, "positions") == 0) {
    return run_positions(model, options);
  }
  if (std::strcmp(options.workload, "pointer-swaps") == 0) {
    return run_pointer_swaps(model, options);
  }
  if (std::strcmp(options.workload, "cancellation") == 0) {
    return run_cancellation(model, options);
  }
  if (std::strcmp(options.workload, "same-math") == 0) {
    return run_same_math(model, options);
  }
  if (std::strcmp(options.workload, "graph-ab") == 0) {
    return run_graph_ab(model, options);
  }
  if (std::strcmp(options.workload, "state-memory") == 0) {
    return run_state_memory(model, options);
  }
  return usage(argv[0]);
}
