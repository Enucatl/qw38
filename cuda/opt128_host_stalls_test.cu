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
#include <numeric>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT128_HOST_STALLS_RESULT=";
constexpr char kCounts[] = "QW38_OPT128_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 16;
constexpr std::size_t kLogitsBytes =
    qw38::internal::kVocabularySize * sizeof(float);

struct Options final {
  const char* workload = "profile";
  const char* candidate = "none";
  std::size_t prefix = 128;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
  std::size_t capacity = 0;
  bool poll = false;
};

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
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
  qw38::cuda::HostStallTimings stalls{};
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
      "usage: %s [--workload profile|lifetime|cancellation|logits-consumer|"
      "non-greedy|stall-ab] [--candidate none|poll|defer|combined] "
      "[--poll 0|1] [--prefix N] [--warmups N] [--samples N] [--tokens N] "
      "[--capacity N] MODEL independently_restored=true same_binary=true\n",
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
    } else if (std::strcmp(arg, "--candidate") == 0 && index + 1 < argc) {
      options->candidate = argv[++index];
    } else if (std::strcmp(arg, "--poll") == 0 && index + 1 < argc) {
      options->poll = std::strcmp(argv[++index], "1") == 0;
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
  return static_cast<float>(std::chrono::duration<double, std::milli>(
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

qw38::Status never_cancel(void*) noexcept { return qw38::Status::ok(); }

qw38::Status poll_stop(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls >= context->stop_after) {
    return {qw38::StatusCode::kCancelled, "opt128 host stall cancellation"};
  }
  return qw38::Status::ok();
}

void apply_candidate(const char* candidate) {
  const bool poll = std::strcmp(candidate, "poll") == 0 ||
                    std::strcmp(candidate, "combined") == 0;
  const bool defer = std::strcmp(candidate, "defer") == 0 ||
                     std::strcmp(candidate, "combined") == 0;
  qw38::cuda::set_poll_without_device_sync_override(poll);
  qw38::cuda::set_defer_elapsed_event_sync_override(defer);
}

void clear_candidate() {
  qw38::cuda::clear_poll_without_device_sync_override();
  qw38::cuda::clear_defer_elapsed_event_sync_override();
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

qw38::Status prefill(const qw38::cuda::ResidentModel& model,
                     const std::vector<std::size_t>& tokens, std::size_t prefix,
                     qw38::cuda::SchedulerSession* session,
                     qw38::cuda::SchedulerWorkspace* workspace,
                     qw38::cuda::SchedulerGraphs* graphs, float* logits,
                     float* hidden) {
  qw38::cuda::SyncResult sync{};
  return qw38::cuda::sync_tokens(model, tokens.data(), prefix, session,
                                 workspace, logits,
                                 qw38::internal::kVocabularySize, hidden,
                                 qw38::internal::kResidualWidth, &sync, nullptr,
                                 graphs);
}

qw38::Status create_graphs(const qw38::cuda::ResidentModel& model,
                           qw38::cuda::SchedulerSession* session,
                           qw38::cuda::SchedulerWorkspace* workspace,
                           qw38::cuda::SchedulerGraphs* graphs) {
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  return graphs->create(model, workspace, session);
}

void print_counts(std::size_t warmups, std::size_t samples,
                  std::size_t candidates, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-128\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

void print_stalls(const char* name, const qw38::cuda::HostStallTimings& stalls,
                  std::size_t tokens) {
  const float scale = tokens == 0 ? 1.0F : static_cast<float>(tokens);
  std::printf(
      "\"%s\":{\"tokens\":%zu,\"wall_ms\":%.9g,\"event_create_ms\":%.9g,"
      "\"event_destroy_ms\":%.9g,\"elapsed_event_sync_ms\":%.9g,"
      "\"scatter_device_sync_ms\":%.9g,\"finish_output_commit_ms\":%.9g,"
      "\"greedy_d2h_ms\":%.9g,\"graph_launch_host_ms\":%.9g,"
      "\"poll_device_sync_ms\":%.9g,\"poll_host_ms\":%.9g,"
      "\"launch_param_update_ms\":%.9g,\"sampling_ms\":%.9g,"
      "\"overlapping_wait_ms\":%.9g,\"delaying_host_ms\":%.9g,"
      "\"per_token_wall_ms\":%.9g,\"per_token_delaying_ms\":%.9g,"
      "\"poll_device_syncs\":%u,\"poll_calls\":%u,\"graph_launches\":%u,"
      "\"blocking_d2h_copies\":%u,\"poll_without_device_sync\":%s,"
      "\"defer_elapsed_event_sync\":%s}",
      name, tokens, static_cast<double>(stalls.wall_ms),
      static_cast<double>(stalls.event_create_ms),
      static_cast<double>(stalls.event_destroy_ms),
      static_cast<double>(stalls.elapsed_event_sync_ms),
      static_cast<double>(stalls.scatter_device_sync_ms),
      static_cast<double>(stalls.finish_output_commit_ms),
      static_cast<double>(stalls.greedy_d2h_ms),
      static_cast<double>(stalls.graph_launch_host_ms),
      static_cast<double>(stalls.poll_device_sync_ms),
      static_cast<double>(stalls.poll_host_ms),
      static_cast<double>(stalls.launch_param_update_ms),
      static_cast<double>(stalls.sampling_ms),
      static_cast<double>(stalls.overlapping_wait_ms),
      static_cast<double>(stalls.delaying_host_ms),
      static_cast<double>(stalls.wall_ms / scale),
      static_cast<double>(stalls.delaying_host_ms / scale),
      stalls.poll_device_syncs, stalls.poll_calls, stalls.graph_launches,
      stalls.blocking_d2h_copies, json_bool(stalls.poll_without_device_sync),
      json_bool(stalls.defer_elapsed_event_sync));
}

qw38::Status run_sample_eval_loop(
    const qw38::cuda::ResidentModel& model, const std::vector<std::size_t>& tokens,
    std::size_t prefix, std::size_t outputs, bool use_poll, bool materialize,
    qw38::cuda::HostStallTimings* stalls, float* decode_ms, float* p50,
    float* p95, std::uint64_t* d2h_bytes, bool* greedy_ok) {
  const std::size_t capacity = session_capacity(prefix, outputs, 0);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &session, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  workspace.reset_transfer_counters();
  qw38::cuda::EvalControl control{};
  control.poll = never_cancel;
  const qw38::cuda::EvalControl* control_pointer = use_poll ? &control : nullptr;
  std::vector<float> latencies;
  latencies.reserve(outputs);
  qw38::cuda::HostStallTimings local{};
  qw38::cuda::HostStallScope stall_scope(&local);
  qw38::cuda::ExecutionGraphPathScope path_scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  const auto decode_started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    std::size_t sampled = 0;
    const auto token_started = std::chrono::steady_clock::now();
    status = qw38::cuda::greedy_sample(session, &sampled);
    float elapsed = 0.0F;
    if (status.is_ok()) {
      status = run_token(model, sampled, &session, &workspace, logits.data(),
                         hidden.data(), &elapsed, &graphs, control_pointer);
    }
    if (status.is_ok() && materialize) {
      status = session.copy_last_outputs(logits.data(), logits.size(),
                                         hidden.data(), hidden.size());
    }
    latencies.push_back(wall_ms(token_started));
  }
  local.wall_ms = wall_ms(decode_started);
  if (stalls != nullptr) *stalls = local;
  if (decode_ms != nullptr) *decode_ms = local.wall_ms;
  if (p50 != nullptr) *p50 = percentile(latencies, 0.50F);
  if (p95 != nullptr) *p95 = percentile(latencies, 0.95F);
  if (d2h_bytes != nullptr) *d2h_bytes = workspace.transfer_d2h_bytes_;
  if (greedy_ok != nullptr) *greedy_ok = session.greedy_token_valid();
  return status;
}

int run_profile(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 16 : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  struct Arm final {
    const char* name;
    const char* candidate;
    bool poll;
  };
  const Arm arms[] = {
      {"parent_no_poll", "none", false},
      {"parent_server_poll", "none", true},
      {"poll_host_only", "poll", true},
      {"defer_elapsed_sync", "defer", false},
      {"combined_server_poll", "combined", true},
  };
  bool all_ok = true;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-128\",\"workload\":\"profile\","
      "\"parent\":\"decode_segments8\",\"prefix\":%zu,\"outputs\":%zu,"
      "\"lazy_logits\":true,\"greedy_transfer_bytes\":4,\"arms\":{",
      kPrefix, prefix, outputs);
  for (int index = 0; index < 5; ++index) {
    apply_candidate(arms[index].candidate);
    qw38::cuda::HostStallTimings stalls{};
    float decode_ms = 0.0F;
    float p50 = 0.0F;
    float p95 = 0.0F;
    std::uint64_t d2h = 0;
    bool greedy_ok = false;
    const qw38::Status status = run_sample_eval_loop(
        model, tokens, prefix, outputs, arms[index].poll, false, &stalls,
        &decode_ms, &p50, &p95, &d2h, &greedy_ok);
    const bool ok = status.is_ok() && greedy_ok && d2h == 4 * outputs;
    all_ok = all_ok && ok;
    if (index > 0) std::printf(",");
    std::printf("\"%s\":{\"ok\":%s,\"d2h_bytes\":%llu,\"greedy_ok\":%s,\"p50_ms\":%.9g,"
                "\"p95_ms\":%.9g,",
                arms[index].name, json_bool(ok),
                static_cast<unsigned long long>(d2h), json_bool(greedy_ok),
                static_cast<double>(p50), static_cast<double>(p95));
    print_stalls("stalls", stalls, outputs);
    std::printf("}");
    clear_candidate();
    cudaDeviceSynchronize();
    cudaGetLastError();
  }
  std::printf(
      "},\"ok\":%s,\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":2,\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true}\n",
      json_bool(all_ok));
  print_counts(0, 1, 2, false);
  return all_ok ? 0 : 1;
}

int run_lifetime(const qw38::cuda::ResidentModel& model, const Options& options) {
  apply_candidate(options.candidate);
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 3, options.capacity);
  std::vector<std::size_t> tokens(prefix + 3);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &session, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::ExecutionGraphPathScope path_scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  workspace.reset_transfer_counters();
  std::size_t greedy = 0;
  if (status.is_ok()) status = qw38::cuda::greedy_sample(session, &greedy);
  const bool greedy_before = session.greedy_token_valid();
  std::vector<float> before(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> before_hidden{};
  if (status.is_ok()) {
    status = session.copy_last_outputs(before.data(), before.size(),
                                       before_hidden.data(),
                                       before_hidden.size());
  }
  std::vector<float> again(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> again_hidden{};
  if (status.is_ok()) {
    status = session.copy_last_outputs(again.data(), again.size(),
                                       again_hidden.data(), again_hidden.size());
  }
  const bool repeat_equal =
      std::memcmp(before.data(), again.data(), before.size() * sizeof(float)) ==
      0;
  float elapsed = 0.0F;
  if (status.is_ok()) {
    status = run_token(model, greedy, &session, &workspace, logits.data(),
                       hidden.data(), &elapsed, &graphs);
  }
  const std::uint64_t lazy_d2h = workspace.transfer_d2h_bytes_;
  std::vector<float> after_eval(qw38::internal::kVocabularySize);
  if (status.is_ok()) {
    status = session.copy_last_outputs(after_eval.data(), after_eval.size(),
                                       hidden.data(), hidden.size());
  }
  const bool eval_changed =
      std::memcmp(before.data(), after_eval.data(),
                  before.size() * sizeof(float)) != 0;
  const char* tmp = "/tmp/qw38-opt128-ckpt.bin";
  if (status.is_ok()) status = session.save_checkpoint(tmp);
  qw38::cuda::SchedulerSession restored;
  if (status.is_ok()) status = restored.create(capacity);
  if (status.is_ok()) status = restored.restore_checkpoint(tmp, &workspace);
  bool equal = false;
  if (status.is_ok()) status = restored.state_equals(session, &equal);
  std::remove(tmp);
  PollContext poll{0, 1};
  const qw38::cuda::EvalControl cancel{poll_stop, &poll};
  const std::size_t frontier_before = session.frontier();
  qw38::Status cancelled = run_token(model, tokens[prefix + 1], &session,
                                     &workspace, logits.data(), hidden.data(),
                                     &elapsed, &graphs, &cancel);
  const bool cancel_isolated =
      cancelled.code() == qw38::StatusCode::kCancelled &&
      session.frontier() == frontier_before;
  float next_ms = 0.0F;
  if (status.is_ok()) {
    status = run_token(model, tokens[prefix + 1], &session, &workspace,
                       logits.data(), hidden.data(), &next_ms, &graphs);
  }
  const bool subsequent_ok = status.is_ok() && session.frontier() == frontier_before + 1;
  clear_candidate();
  const bool ok = status.is_ok() && greedy_before && repeat_equal && eval_changed &&
                  equal && cancel_isolated && subsequent_ok && lazy_d2h == 4;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-128\",\"workload\":\"lifetime\","
      "\"ok\":%s,\"greedy_valid\":%s,\"repeat_logits_equal\":%s,"
      "\"eval_changes_logits\":%s,\"restore_equals\":%s,\"cancel_isolated\":%s,"
      "\"subsequent_request_ok\":%s,\"lazy_d2h_bytes\":%llu,"
      "\"full_logit_bytes\":%zu,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(greedy_before), json_bool(repeat_equal),
      json_bool(eval_changed), json_bool(equal), json_bool(cancel_isolated),
      json_bool(subsequent_ok), static_cast<unsigned long long>(lazy_d2h),
      kLogitsBytes);
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_cancellation(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  apply_candidate(options.candidate);
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  bool all_ok = true;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-128\","
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
      status = create_graphs(model, &session, &workspace, &graphs);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    qw38::cuda::ExecutionGraphPathScope path_scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
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
      token_status =
          run_token(model, tokens[prefix], &session, &workspace, logits.data(),
                    hidden.data(), &elapsed, &graphs, &control);
    }
    const bool cancelled = token_status.code() == qw38::StatusCode::kCancelled;
    const bool preserved = session.frontier() == frontier_before;
    const bool ok = status.is_ok() && cancelled && preserved &&
                    poll_context.calls >= 1;
    all_ok = all_ok && ok;
    std::printf(
        "%s{\"name\":\"%s\",\"ok\":%s,\"cancelled\":%s,\"frontier_preserved\":%s,"
        "\"poll_calls\":%zu,\"stop_after\":%zu,\"atomic_commit\":%s}",
        case_index == 0 ? "" : ",", names[case_index], json_bool(ok),
        json_bool(cancelled), json_bool(preserved), poll_context.calls,
        stops[case_index], json_bool(preserved));
  }
  clear_candidate();
  std::printf(
      "],\"ok\":%s,\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false}\n",
      json_bool(all_ok));
  print_counts(0, 1, 1, false);
  return all_ok ? 0 : 1;
}

int run_logits_consumer(const qw38::cuda::ResidentModel& model,
                        const Options& options) {
  apply_candidate(options.candidate);
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 4 : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  qw38::cuda::HostStallTimings lazy{};
  std::uint64_t lazy_d2h = 0;
  bool greedy_ok = false;
  qw38::Status status = run_sample_eval_loop(
      model, tokens, prefix, outputs, false, false, &lazy, nullptr, nullptr,
      nullptr, &lazy_d2h, &greedy_ok);
  const std::size_t capacity = session_capacity(prefix, outputs, 0);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &session, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::ExecutionGraphPathScope path_scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  std::size_t greedy = 0;
  std::size_t from_logits = 0;
  bool logits_finite = false;
  if (status.is_ok()) status = qw38::cuda::greedy_sample(session, &greedy);
  if (status.is_ok()) {
    status = session.copy_last_outputs(logits.data(), logits.size(),
                                       hidden.data(), hidden.size());
  }
  if (status.is_ok()) {
    logits_finite = true;
    from_logits = 0;
    for (std::size_t index = 1; index < logits.size(); ++index) {
      if (!(logits[index] == logits[index])) {
        logits_finite = false;
        break;
      }
      if (logits[index] > logits[from_logits]) from_logits = index;
    }
  }
  clear_candidate();
  const bool ok = status.is_ok() && greedy_ok && lazy_d2h == 4 * outputs &&
                  logits_finite && from_logits == greedy;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-128\","
      "\"workload\":\"logits-consumer\",\"ok\":%s,\"lazy_d2h_bytes\":%llu,"
      "\"full_logit_bytes\":%zu,\"greedy_ok\":%s,\"logits_finite\":%s,"
      "\"greedy_from_full_logits\":%zu,\"cached_greedy\":%zu,\"keep\":false}\n",
      kPrefix, json_bool(ok), static_cast<unsigned long long>(lazy_d2h),
      kLogitsBytes, json_bool(greedy_ok), json_bool(logits_finite), from_logits,
      greedy);
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_non_greedy(const qw38::cuda::ResidentModel& model,
                   const Options& options) {
  apply_candidate(options.candidate);
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 1, options.capacity);
  std::vector<std::size_t> tokens(prefix + 1);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &session, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::ExecutionGraphPathScope path_scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  const auto materialize_started = std::chrono::steady_clock::now();
  if (status.is_ok()) {
    status = session.copy_last_outputs(logits.data(), logits.size(),
                                       hidden.data(), hidden.size());
  }
  const float materialize_ms = wall_ms(materialize_started);
  const auto alloc_started = std::chrono::steady_clock::now();
  std::vector<std::size_t> candidates(qw38::internal::kVocabularySize);
  std::iota(candidates.begin(), candidates.end(), 0);
  std::vector<double> weights(candidates.size());
  const float alloc_ms = wall_ms(alloc_started);
  clear_candidate();
  const bool ok = status.is_ok() && !logits.empty();
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-128\",\"workload\":\"non-greedy\","
      "\"ok\":%s,\"materialize_ms\":%.9g,\"host_alloc_ms\":%.9g,"
      "\"candidate_count\":%zu,\"keep\":false}\n",
      kPrefix, json_bool(ok), static_cast<double>(materialize_ms),
      static_cast<double>(alloc_ms), candidates.size());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

qw38::Status run_request_arm(const qw38::cuda::ResidentModel& model,
                             const std::vector<std::size_t>& tokens,
                             std::size_t prefix, std::size_t outputs,
                             const char* candidate, bool poll,
                             ArmResult* out) {
  apply_candidate(candidate);
  const auto setup_started = std::chrono::steady_clock::now();
  const std::size_t capacity = session_capacity(prefix, outputs, 0);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &session, &workspace, &graphs);
  out->setup_ms = wall_ms(setup_started);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  const auto request_started = std::chrono::steady_clock::now();
  const auto prefill_started = std::chrono::steady_clock::now();
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  out->prefill_ms = wall_ms(prefill_started);
  out->ttft_ms = out->prefill_ms;
  qw38::cuda::EvalControl control{};
  control.poll = never_cancel;
  const qw38::cuda::EvalControl* control_pointer = poll ? &control : nullptr;
  std::vector<float> latencies;
  latencies.reserve(outputs);
  qw38::cuda::HostStallTimings stalls{};
  qw38::cuda::HostStallScope stall_scope(&stalls);
  qw38::cuda::ExecutionGraphPathScope path_scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  const auto decode_started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    std::size_t sampled = 0;
    const auto token_started = std::chrono::steady_clock::now();
    status = qw38::cuda::greedy_sample(session, &sampled);
    float elapsed = 0.0F;
    if (status.is_ok()) {
      status = run_token(model, sampled, &session, &workspace, logits.data(),
                         hidden.data(), &elapsed, &graphs, control_pointer);
    }
    latencies.push_back(wall_ms(token_started));
  }
  stalls.wall_ms = wall_ms(decode_started);
  out->decode_only_ms = stalls.wall_ms;
  out->request_ms = out->setup_ms + wall_ms(request_started);
  out->stalls = stalls;
  const std::size_t counted = outputs == 0 ? prefix : outputs;
  out->request_tok_s =
      out->request_ms > 0.0F
          ? static_cast<float>(counted) * 1000.0F / out->request_ms
          : 0.0F;
  out->decode_only_tok_s =
      outputs > 0 && out->decode_only_ms > 0.0F
          ? static_cast<float>(outputs) * 1000.0F / out->decode_only_ms
          : out->request_tok_s;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  clear_candidate();
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"setup_ms\":%.9g,\"prefill_ms\":%.9g,\"decode_only_ms\":%.9g,"
      "\"request_ms\":%.9g,\"ttft_ms\":%.9g,\"decode_only_tok_s\":%.9g,"
      "\"request_tok_s\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,",
      name, static_cast<double>(arm.setup_ms), static_cast<double>(arm.prefill_ms),
      static_cast<double>(arm.decode_only_ms),
      static_cast<double>(arm.request_ms), static_cast<double>(arm.ttft_ms),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.request_tok_s),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms));
  print_stalls("stalls", arm.stalls,
               arm.decode_only_ms > 0.0F ? 1 : 1);
  std::printf("}%s", last ? "" : ",");
}

int run_stall_ab(const qw38::cuda::ResidentModel& model, const Options& options) {
  const bool prefill_only = options.tokens == 0;
  const std::size_t prompt = options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + std::max(outputs, std::size_t{1}));
  fill_tokens(&tokens);
  const char* candidate = options.candidate;
  const bool poll = options.poll || std::strcmp(candidate, "poll") == 0 ||
                    std::strcmp(candidate, "combined") == 0;
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_request_arm(model, tokens, prompt, outputs, "none", poll,
                             &discarded);
    if (status.is_ok()) {
      status = run_request_arm(model, tokens, prompt, outputs, candidate, poll,
                               &discarded);
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
  for (std::size_t sample = 0; sample < options.samples; ++sample) {
    const bool ba = (sample % 2) == 1;
    ArmResult first;
    ArmResult second;
    status = run_request_arm(model, tokens, prompt, outputs,
                             ba ? candidate : "none", poll, &first);
    if (status.is_ok()) {
      status = run_request_arm(model, tokens, prompt, outputs,
                               ba ? "none" : candidate, poll, &second);
    }
    if (!status.is_ok()) return fail_status(status);
    arm_a[sample] = ba ? second : first;
    arm_b[sample] = ba ? first : second;
    std::printf(
        "quartz_pair sample_index=%zu order=%s A_decode_only_tok_s=%.9g "
        "B_decode_only_tok_s=%.9g A_request_tok_s=%.9g B_request_tok_s=%.9g "
        "independently_restored=true same_binary=true\n",
        sample, ba ? "BA" : "AB",
        static_cast<double>(arm_a[sample].decode_only_tok_s),
        static_cast<double>(arm_b[sample].decode_only_tok_s),
        static_cast<double>(arm_a[sample].request_tok_s),
        static_cast<double>(arm_b[sample].request_tok_s));
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-128\",\"workload\":\"stall-ab\","
      "\"candidate\":\"%s\",\"poll\":%s,\"prefix\":%zu,\"prefill\":%s,"
      "\"warmups\":%zu,\"samples\":%zu,\"tokens\":%zu,"
      "\"parent\":\"decode_segments8\",\"metrics\":[\"decode_only\","
      "\"complete_request\"],\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true,\"pairs\":[",
      kPrefix, candidate, json_bool(poll), prompt, json_bool(prefill_only),
      options.warmups, options.samples, outputs, options.warmups,
      options.samples);
  for (std::size_t sample = 0; sample < options.samples; ++sample) {
    if (sample != 0) std::printf(",");
    std::printf("{\"sample_index\":%zu,\"order\":\"%s\",", sample,
                (sample % 2) == 1 ? "BA" : "AB");
    print_arm("A", arm_a[sample], false);
    print_arm("B", arm_b[sample], true);
    std::printf("}");
  }
  std::printf("]}\n");
  print_counts(options.warmups, options.samples, 2, false);
  return 0;
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
  if (std::strcmp(options.workload, "profile") == 0) {
    return run_profile(model, options);
  }
  if (std::strcmp(options.workload, "lifetime") == 0) {
    return run_lifetime(model, options);
  }
  if (std::strcmp(options.workload, "cancellation") == 0) {
    return run_cancellation(model, options);
  }
  if (std::strcmp(options.workload, "logits-consumer") == 0) {
    return run_logits_consumer(model, options);
  }
  if (std::strcmp(options.workload, "non-greedy") == 0) {
    return run_non_greedy(model, options);
  }
  if (std::strcmp(options.workload, "stall-ab") == 0) {
    return run_stall_ab(model, options);
  }
  return usage(argv[0]);
}
