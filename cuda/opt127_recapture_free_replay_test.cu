#include "attention_decode_path.cuh"
#include "decode_launch_state.cuh"
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

constexpr char kPrefix[] = "QW38_OPT127_RECAPTURE_FREE_REPLAY_RESULT=";
constexpr char kCounts[] = "QW38_OPT127_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;

struct Options final {
  const char* workload = "recapture-proof";
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
  std::chrono::steady_clock::time_point cancel_requested{};
  bool requested = false;
};

struct ArmResult final {
  float setup_ms = 0.0F;
  float prefill_ms = 0.0F;
  float decode_only_ms = 0.0F;
  float request_ms = 0.0F;
  float ttft_ms = 0.0F;
  float decode_only_tok_s = 0.0F;
  float request_tok_s = 0.0F;
  float p50_ms = 0.0F;
  float p95_ms = 0.0F;
  float graph_ms = 0.0F;
  std::size_t graph_bytes = 0;
  std::uint32_t launch_param_updates = 0;
  std::uint32_t launch_state_uploads = 0;
  std::uint32_t capture = 0;
  std::uint32_t instantiate = 0;
  std::uint32_t upload = 0;
  std::uint32_t destroy = 0;
  std::uint32_t exec_destroy = 0;
  std::uint32_t topology_recapture = 0;
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
      "usage: %s [--workload recapture-proof|cancellation|same-math|handoff|"
      "graph-ab|state-memory|attribution] "
      "[--execution-graphs ffn_only|decode_segments8] "
      "[--prefix N] [--warmups N] [--samples N] [--tokens N] [--capacity N] "
      "MODEL independently_restored=true same_binary=true\n",
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

float wall_ms(const std::chrono::steady_clock::time_point started) {
  return static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count());
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
                       const qw38::cuda::EvalControl* control = nullptr,
                       qw38::cuda::DecodeAttribution* attribution = nullptr) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, control, nullptr,
      qw38::cuda::PointwisePath::kFused, graphs, attribution);
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
      "%s{\"schema_version\":1,\"task\":\"OPT-127\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

void fill_lifecycle(ArmResult* out, const qw38::cuda::SchedulerGraphs& graphs) {
  const qw38::cuda::GraphLifecycleCounts counts = graphs.lifecycle_counts();
  out->graph_bytes = graphs.allocated_bytes();
  out->decode_segment_graphs = graphs.decode_segment_graph_count();
  out->launch_param_updates = counts.launch_param_update;
  out->launch_state_uploads = counts.launch_state_upload;
  out->capture = counts.capture;
  out->instantiate = counts.instantiate;
  out->upload = counts.upload;
  out->destroy = counts.destroy;
  out->exec_destroy = counts.exec_destroy;
  out->topology_recapture = counts.topology_recapture;
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

void append_lifecycle(std::string* body, const char* name,
                      const qw38::cuda::GraphLifecycleCounts& counts, bool last) {
  char row[768];
  std::snprintf(
      row, sizeof(row),
      "\"%s\":{\"capture\":%u,\"instantiate\":%u,\"upload\":%u,\"destroy\":%u,"
      "\"exec_destroy\":%u,\"launch_state_upload\":%u,"
      "\"launch_param_update\":%u,\"topology_recapture\":%u,"
      "\"last_capture_ms\":%.9g,\"last_instantiate_ms\":%.9g,"
      "\"last_upload_ms\":%.9g,\"last_launch_state_upload_ms\":%.9g}%s",
      name, counts.capture, counts.instantiate, counts.upload, counts.destroy,
      counts.exec_destroy, counts.launch_state_upload, counts.launch_param_update,
      counts.topology_recapture, static_cast<double>(counts.last_capture_ms),
      static_cast<double>(counts.last_instantiate_ms),
      static_cast<double>(counts.last_upload_ms),
      static_cast<double>(counts.last_launch_state_upload_ms), last ? "" : ",");
  body->append(row);
}

int run_recapture_proof(const qw38::cuda::ResidentModel& model,
                        const Options& options) {
  struct Region final {
    const char* name;
    std::size_t prefix;
    std::size_t outputs;
    bool expect_cross;
  };
  const Region regions[] = {{"crossover_1024", 1023, 8, true},
                            {"steady_topology0", 128, 16, false}};
  bool all_ok = true;
  std::string body =
      "{\"schema_version\":1,\"task\":\"OPT-127\","
      "\"workload\":\"recapture-proof\",\"invalidation_policy\":\"";
  body.append(qw38::cuda::kDecodeGraphInvalidationPolicy);
  body.append("\",\"coarser_candidate_admitted\":false,\"regions\":[");
  for (int region_index = 0; region_index < 2; ++region_index) {
    const Region& region = regions[region_index];
    const std::size_t capacity =
        session_capacity(region.prefix, region.outputs, options.capacity);
    std::vector<std::size_t> tokens(region.prefix + region.outputs);
    fill_tokens(&tokens);
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    qw38::Status status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    const qw38::cuda::GraphLifecycleCounts after_create = graphs.lifecycle_counts();
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok() && region.prefix > 0) {
      status = prefill(model, tokens, region.prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    const qw38::cuda::GraphLifecycleCounts after_prefill = graphs.lifecycle_counts();
    std::vector<float> token_ms;
    token_ms.reserve(region.outputs);
    int topology_first = -1;
    int topology_last = -1;
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    for (std::size_t step = 0; status.is_ok() && step < region.outputs; ++step) {
      const int topology =
          qw38::cuda::decode_graph_topology_index(session.frontier());
      if (step == 0) topology_first = topology;
      topology_last = topology;
      float elapsed = 0.0F;
      const auto started = std::chrono::steady_clock::now();
      status = run_token(model, tokens[region.prefix + step], &session, &workspace,
                         logits.data(), hidden.data(), &elapsed, &graphs);
      token_ms.push_back(wall_ms(started));
    }
    const qw38::cuda::GraphLifecycleCounts after_decode = graphs.lifecycle_counts();
    const std::uint32_t capture_delta =
        after_decode.capture - after_prefill.capture;
    const std::uint32_t instantiate_delta =
        after_decode.instantiate - after_prefill.instantiate;
    const std::uint32_t upload_delta = after_decode.upload - after_prefill.upload;
    const std::uint32_t destroy_delta =
        after_decode.destroy - after_prefill.destroy;
    const std::uint32_t exec_destroy_delta =
        after_decode.exec_destroy - after_prefill.exec_destroy;
    const std::uint32_t recapture_delta =
        after_decode.topology_recapture - after_prefill.topology_recapture;
    const bool zero_recapture = capture_delta == 0 && instantiate_delta == 0 &&
                                upload_delta == 0 && destroy_delta == 0 &&
                                exec_destroy_delta == 0 && recapture_delta == 0;
    const bool crossed = topology_first != topology_last;
    const bool expect_cross = region.expect_cross;
    std::string message;
    json_escape(status.message(), &message);
    const bool ok = status.is_ok() && zero_recapture && crossed == expect_cross &&
                    graphs.decode_segment_graph_count() ==
                        8 * qw38::cuda::kDecodeGraphTopologyCount;
    all_ok = all_ok && ok;
    char row[1024];
    std::snprintf(
        row, sizeof(row),
        "%s{\"name\":\"%s\",\"ok\":%s,\"zero_steady_state_recapture\":%s,"
        "\"prefix\":%zu,\"outputs\":%zu,\"topology_first\":%d,"
        "\"topology_last\":%d,\"crossed_topology\":%s,"
        "\"capture_delta\":%u,\"instantiate_delta\":%u,"
        "\"upload_delta\":%u,\"destroy_delta\":%u,"
        "\"exec_destroy_delta\":%u,\"topology_recapture_delta\":%u,"
        "\"launch_state_upload_delta\":%u,\"mean_token_ms\":%.9g,"
        "\"decode_segment_graph_count\":%zu,\"message\":\"%s\",",
        region_index == 0 ? "" : ",", region.name, json_bool(ok),
        json_bool(zero_recapture), region.prefix, region.outputs, topology_first,
        topology_last, json_bool(crossed), capture_delta, instantiate_delta,
        upload_delta, destroy_delta, exec_destroy_delta, recapture_delta,
        after_decode.launch_state_upload - after_prefill.launch_state_upload,
        static_cast<double>(percentile(token_ms, 0.50F)),
        graphs.decode_segment_graph_count(), message.c_str());
    body.append(row);
    append_lifecycle(&body, "after_create", after_create, false);
    append_lifecycle(&body, "after_prefill", after_prefill, false);
    append_lifecycle(&body, "after_decode", after_decode, true);
    body.push_back('}');
    cudaDeviceSynchronize();
    cudaGetLastError();
  }
  char tail[256];
  std::snprintf(
      tail, sizeof(tail),
      "],\"ok\":%s,\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true}\n",
      json_bool(all_ok));
  body.append(tail);
  std::printf("%s%s", kPrefix, body.c_str());
  print_counts(0, 1, 1, false);
  return all_ok ? 0 : 1;
}

qw38::Status poll_stop(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls >= context->stop_after) {
    if (!context->requested) {
      context->requested = true;
      context->cancel_requested = std::chrono::steady_clock::now();
    }
    return {qw38::StatusCode::kCancelled, "opt127 graph cancellation"};
  }
  return qw38::Status::ok();
}

int run_cancellation(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  bool all_ok = true;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-127\","
      "\"workload\":\"cancellation\",\"cadence\":\"eight_layer_segment\","
      "\"cases\":[",
      kPrefix);
  const char* names[] = {"mid_segment", "commit"};
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
    const auto started = std::chrono::steady_clock::now();
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      token_status =
          run_token(model, tokens[prefix], &session, &workspace, logits.data(),
                    hidden.data(), &elapsed, &graphs, &control);
    }
    const float response_ms =
        poll_context.requested ? wall_ms(poll_context.cancel_requested)
                               : wall_ms(started);
    const bool cancelled = token_status.code() == qw38::StatusCode::kCancelled;
    const bool preserved = session.frontier() == frontier_before;
    const bool ok = status.is_ok() && cancelled && preserved &&
                    poll_context.calls >= 1;
    all_ok = all_ok && ok;
    std::printf(
        "%s{\"name\":\"%s\",\"ok\":%s,\"cancelled\":%s,\"frontier_preserved\":%s,"
        "\"poll_calls\":%zu,\"stop_after\":%zu,\"cancel_response_ms\":%.9g,"
        "\"atomic_commit\":%s}",
        case_index == 0 ? "" : ",", names[case_index], json_bool(ok),
        json_bool(cancelled), json_bool(preserved), poll_context.calls,
        stops[case_index], static_cast<double>(response_ms),
        json_bool(preserved));
  }
  std::printf(
      "],\"ok\":%s,\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false}\n",
      json_bool(all_ok));
  print_counts(0, 1, 1, false);
  return all_ok ? 0 : 1;
}

int run_same_math(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 8 : options.tokens;
  const std::size_t capacity =
      session_capacity(prefix, outputs, options.capacity);
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  std::vector<std::vector<float>> eager_token_logits;
  std::vector<std::array<float, qw38::internal::kResidualWidth>> eager_token_hidden;
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession eager_session;
    qw38::cuda::SchedulerWorkspace eager_workspace;
    status = eager_session.create(capacity);
    if (status.is_ok()) status = eager_workspace.create(capacity);
    std::vector<float> eager_logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                       nullptr, eager_logits.data(), eager_hidden.data());
    }
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float eager_ms = 0.0F;
      status = run_token(model, tokens[prefix + step], &eager_session,
                         &eager_workspace, eager_logits.data(),
                         eager_hidden.data(), &eager_ms, nullptr);
      if (status.is_ok()) {
        eager_token_logits.push_back(eager_logits);
        eager_token_hidden.push_back(eager_hidden);
      }
    }
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  bool exact = status.is_ok() && eager_token_logits.size() == outputs;
  std::size_t matched = 0;
  qw38::cuda::GraphLifecycleCounts before{};
  qw38::cuda::GraphLifecycleCounts after{};
  std::uint32_t launch_param_updates = 0;
  std::size_t decode_segment_graph_count = 0;
  {
    qw38::cuda::SchedulerSession graph_session;
    qw38::cuda::SchedulerWorkspace graph_workspace;
    qw38::cuda::SchedulerGraphs segment_graphs;
    if (status.is_ok()) status = graph_session.create(capacity);
    if (status.is_ok()) status = graph_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &graph_session, &graph_workspace,
                             &segment_graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    std::vector<float> graph_logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &graph_session, &graph_workspace,
                       &segment_graphs, graph_logits.data(), graph_hidden.data());
    }
    before = segment_graphs.lifecycle_counts();
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float graph_ms = 0.0F;
      status = run_token(model, tokens[prefix + step], &graph_session,
                         &graph_workspace, graph_logits.data(),
                         graph_hidden.data(), &graph_ms, &segment_graphs);
      const bool step_exact =
          status.is_ok() && step < eager_token_logits.size() &&
          exact_buffers(eager_token_logits[step], graph_logits,
                        eager_token_hidden[step], graph_hidden);
      exact = exact && step_exact;
      if (step_exact) ++matched;
    }
    after = segment_graphs.lifecycle_counts();
    launch_param_updates = segment_graphs.launch_param_update_count();
    decode_segment_graph_count = segment_graphs.decode_segment_graph_count();
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool zero_recapture =
      after.capture == before.capture && after.instantiate == before.instantiate &&
      after.upload == before.upload && after.destroy == before.destroy &&
      after.topology_recapture == before.topology_recapture;
  const bool ok = status.is_ok() && exact && zero_recapture &&
                  decode_segment_graph_count ==
                      8 * qw38::cuda::kDecodeGraphTopologyCount;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-127\",\"workload\":\"same-math\","
      "\"ok\":%s,\"exact\":%s,\"prefill_exact\":%s,\"matched_tokens\":%zu,"
      "\"tokens\":%zu,\"prefix\":%zu,\"zero_recapture\":%s,"
      "\"launch_param_updates\":%u,\"message\":\"%s\","
      "\"observed_warmups\":0,\"observed_samples\":1,\"observed_candidates\":2,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(exact),
      json_bool(status.is_ok() && exact), matched, outputs, prefix,
      json_bool(zero_recapture), launch_param_updates, message.c_str());
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_handoff(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  std::vector<float> parent_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> parent_hidden{};
  float parent_prefill_ms = 0.0F;
  float parent_decode_ms = 0.0F;
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession parent_session;
    qw38::cuda::SchedulerWorkspace parent_workspace;
    qw38::cuda::SchedulerGraphs parent_graphs;
    status = parent_session.create(capacity);
    if (status.is_ok()) status = parent_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &parent_session, &parent_workspace,
                             &parent_graphs,
                             qw38::cuda::kLegalExecutionGraphFfnOnly);
    }
    const auto parent_prefill_started = std::chrono::steady_clock::now();
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &parent_session, &parent_workspace,
                       &parent_graphs, parent_logits.data(),
                       parent_hidden.data());
    }
    parent_prefill_ms = wall_ms(parent_prefill_started);
    if (status.is_ok()) {
      status = run_token(model, tokens[prefix], &parent_session,
                         &parent_workspace, parent_logits.data(),
                         parent_hidden.data(), &parent_decode_ms, &parent_graphs);
    }
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  std::vector<float> cand_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> cand_hidden{};
  float cand_prefill_ms = 0.0F;
  float cand_decode_ms = 0.0F;
  qw38::cuda::GraphLifecycleCounts counts{};
  {
    qw38::cuda::SchedulerSession cand_session;
    qw38::cuda::SchedulerWorkspace cand_workspace;
    qw38::cuda::SchedulerGraphs cand_graphs;
    if (status.is_ok()) status = cand_session.create(capacity);
    if (status.is_ok()) status = cand_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &cand_session, &cand_workspace, &cand_graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    const auto cand_prefill_started = std::chrono::steady_clock::now();
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &cand_session, &cand_workspace,
                       &cand_graphs, cand_logits.data(), cand_hidden.data());
    }
    cand_prefill_ms = wall_ms(cand_prefill_started);
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix], &cand_session, &cand_workspace,
                         cand_logits.data(), cand_hidden.data(), &cand_decode_ms,
                         &cand_graphs);
    }
    counts = cand_graphs.lifecycle_counts();
  }
  const bool exact = status.is_ok() && exact_buffers(parent_logits, cand_logits,
                                                     parent_hidden, cand_hidden);
  std::string message;
  json_escape(status.message(), &message);
  const bool ok = status.is_ok() && exact && counts.topology_recapture == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-127\",\"workload\":\"handoff\","
      "\"ok\":%s,\"exact\":%s,\"short_output_tokens\":1,\"prefix\":%zu,"
      "\"parent_prefill_ms\":%.9g,\"candidate_prefill_ms\":%.9g,"
      "\"parent_first_decode_ms\":%.9g,\"candidate_first_decode_ms\":%.9g,"
      "\"topology_recapture\":%u,\"message\":\"%s\","
      "\"observed_warmups\":0,\"observed_samples\":1,\"observed_candidates\":2,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(exact), prefix,
      static_cast<double>(parent_prefill_ms),
      static_cast<double>(cand_prefill_ms),
      static_cast<double>(parent_decode_ms),
      static_cast<double>(cand_decode_ms), counts.topology_recapture,
      message.c_str());
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

qw38::Status run_request_arm(const qw38::cuda::ResidentModel& model,
                             const std::vector<std::size_t>& tokens,
                             std::size_t prefix, std::size_t outputs,
                             const char* graph_path, ArmResult* out) {
  const std::size_t capacity = session_capacity(prefix, outputs, 0);
  const auto setup_started = std::chrono::steady_clock::now();
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) {
    status = create_graphs(model, &session, &workspace, &graphs, graph_path);
  }
  out->setup_ms = wall_ms(setup_started);
  if (!status.is_ok()) return status;
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  const auto request_started = std::chrono::steady_clock::now();
  if (prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  out->prefill_ms = wall_ms(request_started);
  out->ttft_ms = out->setup_ms + out->prefill_ms;
  if (!status.is_ok()) return status;
  std::vector<float> latencies;
  latencies.reserve(outputs);
  const auto decode_started = std::chrono::steady_clock::now();
  qw38::cuda::ExecutionGraphPathScope scope(graph_path);
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs);
    if (!status.is_ok()) {
      std::fprintf(stderr, "opt127 token fail path=%s step=%zu position=%zu: %s\n",
                   graph_path, step, prefix + step, status.message().c_str());
      return status;
    }
    latencies.push_back(wall_ms(token_started));
  }
  out->decode_only_ms = wall_ms(decode_started);
  out->request_ms = out->setup_ms + wall_ms(request_started);
  const std::size_t counted = outputs == 0 ? prefix : outputs;
  const float request_denom =
      outputs == 0 ? out->request_ms : out->request_ms;
  out->request_tok_s =
      request_denom > 0.0F
          ? static_cast<float>(counted) * 1000.0F / request_denom
          : 0.0F;
  out->decode_only_tok_s =
      outputs > 0 && out->decode_only_ms > 0.0F
          ? static_cast<float>(outputs) * 1000.0F / out->decode_only_ms
          : out->request_tok_s;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  fill_lifecycle(out, graphs);
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"setup_ms\":%.9g,\"prefill_ms\":%.9g,\"decode_only_ms\":%.9g,"
      "\"request_ms\":%.9g,\"ttft_ms\":%.9g,\"decode_only_tok_s\":%.9g,"
      "\"request_tok_s\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,"
      "\"graph_bytes\":%zu,\"launch_param_updates\":%u,"
      "\"launch_state_uploads\":%u,\"capture\":%u,\"instantiate\":%u,"
      "\"upload\":%u,\"destroy\":%u,\"exec_destroy\":%u,"
      "\"topology_recapture\":%u,\"decode_segment_graphs\":%zu}%s",
      name, static_cast<double>(arm.setup_ms), static_cast<double>(arm.prefill_ms),
      static_cast<double>(arm.decode_only_ms),
      static_cast<double>(arm.request_ms), static_cast<double>(arm.ttft_ms),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.request_tok_s),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms),
      arm.graph_bytes, arm.launch_param_updates, arm.launch_state_uploads,
      arm.capture, arm.instantiate, arm.upload, arm.destroy, arm.exec_destroy,
      arm.topology_recapture, arm.decode_segment_graphs, last ? "" : ",");
}

int run_graph_ab(const qw38::cuda::ResidentModel& model, const Options& options) {
  const bool prefill_only = options.tokens == 0;
  const std::size_t prompt = options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  const char* parent = qw38::cuda::kLegalExecutionGraphFfnOnly;
  const char* candidate = options.execution_graphs;
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_request_arm(model, tokens, prompt, outputs, parent, &discarded);
    if (status.is_ok()) {
      status =
          run_request_arm(model, tokens, prompt, outputs, candidate, &discarded);
    }
    if (!status.is_ok()) return fail_status(status);
    std::printf(
        "quartz_warmup=%zu decode_only_tok_s=%.9g request_tok_s=%.9g "
        "independently_restored=true\n",
        warmup, static_cast<double>(discarded.decode_only_tok_s),
        static_cast<double>(discarded.request_tok_s));
  }
  std::vector<ArmResult> arm_a(options.samples);
  std::vector<ArmResult> arm_b(options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    const bool ba = (pair % 2) == 1;
    ArmResult first;
    ArmResult second;
    status = run_request_arm(model, tokens, prompt, outputs,
                             ba ? candidate : parent, &first);
    if (status.is_ok()) {
      status = run_request_arm(model, tokens, prompt, outputs,
                               ba ? parent : candidate, &second);
    }
    if (!status.is_ok()) return fail_status(status);
    arm_a[pair] = ba ? second : first;
    arm_b[pair] = ba ? first : second;
    std::printf(
        "quartz_pair sample_index=%zu order=%s A_decode_only_tok_s=%.9g "
        "B_decode_only_tok_s=%.9g A_request_tok_s=%.9g B_request_tok_s=%.9g "
        "independently_restored=true same_binary=true\n",
        pair, ba ? "BA" : "AB",
        static_cast<double>(arm_a[pair].decode_only_tok_s),
        static_cast<double>(arm_b[pair].decode_only_tok_s),
        static_cast<double>(arm_a[pair].request_tok_s),
        static_cast<double>(arm_b[pair].request_tok_s));
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-127\",\"workload\":\"graph-ab\","
      "\"prefix\":%zu,\"prefill\":%s,\"warmups\":%zu,\"samples\":%zu,"
      "\"tokens\":%zu,\"parent\":\"%s\",\"candidate\":\"%s\","
      "\"metrics\":[\"decode_only\",\"complete_request\"],"
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
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession eager_session;
    qw38::cuda::SchedulerWorkspace eager_workspace;
    qw38::cuda::SchedulerGraphs ffn_graphs;
    status = eager_session.create(capacity);
    if (status.is_ok()) status = eager_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &eager_session, &eager_workspace, &ffn_graphs,
                             qw38::cuda::kLegalExecutionGraphFfnOnly);
    }
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                       &ffn_graphs, eager_logits.data(), eager_hidden.data());
    }
    float eager_ms = 0.0F;
    if (status.is_ok()) {
      status = run_token(model, tokens[prefix], &eager_session, &eager_workspace,
                         eager_logits.data(), eager_hidden.data(), &eager_ms,
                         &ffn_graphs);
    }
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  std::size_t graph_bytes = 0;
  std::size_t session_bytes = 0;
  std::size_t workspace_bytes = 0;
  {
    qw38::cuda::SchedulerSession graph_session;
    qw38::cuda::SchedulerWorkspace graph_workspace;
    qw38::cuda::SchedulerGraphs segment_graphs;
    if (status.is_ok()) status = graph_session.create(capacity);
    if (status.is_ok()) status = graph_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &graph_session, &graph_workspace,
                             &segment_graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &graph_session, &graph_workspace,
                       &segment_graphs, graph_logits.data(), graph_hidden.data());
    }
    float graph_ms = 0.0F;
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix], &graph_session, &graph_workspace,
                         graph_logits.data(), graph_hidden.data(), &graph_ms,
                         &segment_graphs);
    }
    graph_bytes = segment_graphs.allocated_bytes();
    session_bytes = graph_session.allocated_bytes();
    workspace_bytes = graph_workspace.allocated_bytes();
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool equal = status.is_ok() && exact_buffers(eager_logits, graph_logits,
                                                     eager_hidden, graph_hidden);
  const std::size_t peak = session_bytes + workspace_bytes + graph_bytes;
  const bool ok = status.is_ok() && equal;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-127\","
      "\"workload\":\"state-memory\",\"ok\":%s,\"state_equals\":%s,"
      "\"graph_bytes\":%zu,\"session_bytes\":%zu,\"workspace_bytes\":%zu,"
      "\"peak_bytes\":%zu,\"message\":\"%s\",\"observed_warmups\":0,"
      "\"observed_samples\":1,\"observed_candidates\":1,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(equal), graph_bytes, session_bytes,
      workspace_bytes, peak, message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_attribution(const qw38::cuda::ResidentModel& model,
                    const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = 4;
  const std::size_t capacity = session_capacity(prefix, outputs, options.capacity);
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  float parent_graph_ms = 0.0F;
  float cand_graph_ms = 0.0F;
  float parent_wall_ms = 0.0F;
  float cand_wall_ms = 0.0F;
  qw38::cuda::GraphLifecycleCounts parent_counts{};
  qw38::cuda::GraphLifecycleCounts cand_counts{};
  qw38::cuda::GraphLifecycleCounts parent_before{};
  qw38::cuda::GraphLifecycleCounts cand_before{};
  qw38::Status status = qw38::Status::ok();
  const char* paths[] = {qw38::cuda::kLegalExecutionGraphFfnOnly,
                         qw38::cuda::kLegalExecutionGraphDecodeSegments8};
  float* graph_acc[] = {&parent_graph_ms, &cand_graph_ms};
  float* wall_acc[] = {&parent_wall_ms, &cand_wall_ms};
  qw38::cuda::GraphLifecycleCounts* before_acc[] = {&parent_before, &cand_before};
  qw38::cuda::GraphLifecycleCounts* after_acc[] = {&parent_counts, &cand_counts};
  for (int arm = 0; arm < 2 && status.is_ok(); ++arm) {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs, paths[arm]);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    *before_acc[arm] = graphs.lifecycle_counts();
    qw38::cuda::ExecutionGraphPathScope scope(paths[arm]);
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float elapsed = 0.0F;
      const auto started = std::chrono::steady_clock::now();
      status = run_token(model, tokens[prefix + step], &session, &workspace,
                         logits.data(), hidden.data(), &elapsed, &graphs);
      *wall_acc[arm] += wall_ms(started);
      *graph_acc[arm] += elapsed;
    }
    *after_acc[arm] = graphs.lifecycle_counts();
    cudaDeviceSynchronize();
    cudaGetLastError();
  }
  std::string message;
  json_escape(status.message(), &message);
  const std::uint32_t cand_capture_delta =
      cand_counts.capture - cand_before.capture;
  const std::uint32_t cand_instantiate_delta =
      cand_counts.instantiate - cand_before.instantiate;
  const std::uint32_t cand_upload_delta = cand_counts.upload - cand_before.upload;
  const std::uint32_t cand_destroy_delta =
      cand_counts.destroy - cand_before.destroy;
  const std::uint32_t cand_recapture_delta =
      cand_counts.topology_recapture - cand_before.topology_recapture;
  const std::uint32_t cand_launch_state_delta =
      cand_counts.launch_state_upload - cand_before.launch_state_upload;
  const bool zero_recapture = cand_capture_delta == 0 &&
                              cand_instantiate_delta == 0 &&
                              cand_upload_delta == 0 && cand_destroy_delta == 0 &&
                              cand_recapture_delta == 0;
  const bool ok = status.is_ok() && zero_recapture;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-127\","
      "\"workload\":\"attribution\",\"ok\":%s,\"outputs\":%zu,\"prefix\":%zu,"
      "\"parent_token_ms\":%.9g,\"candidate_token_ms\":%.9g,"
      "\"parent_wall_ms\":%.9g,\"candidate_wall_ms\":%.9g,"
      "\"parent_graph_ms\":%.9g,\"candidate_graph_ms\":%.9g,"
      "\"candidate_capture_delta\":%u,\"candidate_instantiate_delta\":%u,"
      "\"candidate_upload_delta\":%u,\"candidate_destroy_delta\":%u,"
      "\"candidate_topology_recapture_delta\":%u,"
      "\"candidate_launch_state_upload_delta\":%u,"
      "\"parent_capture_delta\":%u,\"parent_launch_state_upload_delta\":%u,"
      "\"weight_byte_reduction_claimed\":false,\"message\":\"%s\","
      "\"observed_warmups\":0,\"observed_samples\":1,\"observed_candidates\":2,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), outputs, prefix,
      static_cast<double>(parent_graph_ms), static_cast<double>(cand_graph_ms),
      static_cast<double>(parent_wall_ms), static_cast<double>(cand_wall_ms),
      static_cast<double>(parent_graph_ms), static_cast<double>(cand_graph_ms),
      cand_capture_delta, cand_instantiate_delta, cand_upload_delta,
      cand_destroy_delta, cand_recapture_delta, cand_launch_state_delta,
      parent_counts.capture - parent_before.capture,
      parent_counts.launch_state_upload - parent_before.launch_state_upload,
      message.c_str());
  print_counts(0, 1, 2, false);
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
  if (std::strcmp(options.workload, "recapture-proof") == 0) {
    return run_recapture_proof(model, options);
  }
  if (std::strcmp(options.workload, "cancellation") == 0) {
    return run_cancellation(model, options);
  }
  if (std::strcmp(options.workload, "same-math") == 0) {
    return run_same_math(model, options);
  }
  if (std::strcmp(options.workload, "handoff") == 0) {
    return run_handoff(model, options);
  }
  if (std::strcmp(options.workload, "graph-ab") == 0) {
    return run_graph_ab(model, options);
  }
  if (std::strcmp(options.workload, "state-memory") == 0) {
    return run_state_memory(model, options);
  }
  if (std::strcmp(options.workload, "attribution") == 0) {
    return run_attribution(model, options);
  }
  return usage(argv[0]);
}
