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

constexpr char kPrefix[] = "QW38_OPT118_TRANSFER_RESIDENCY_RESULT=";
constexpr char kCounts[] = "QW38_OPT118_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;
constexpr std::size_t kLogitsBytes =
    qw38::internal::kVocabularySize * sizeof(float);
constexpr std::size_t kHiddenBytes =
    qw38::internal::kResidualWidth * sizeof(float);

struct Options final {
  const char* workload = "inventory";
  const char* execution_graphs = "ffn_only";
  std::size_t prefix = 128;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
  std::size_t capacity = 0;
  bool materialize_every_token = false;
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
  float ttft_ms = 0.0F;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t d2d_bytes = 0;
  std::uint32_t d2h_copies = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload inventory|greedy-parity|api|"
               "cancellation|state-memory|session-ab] "
               "[--execution-graphs ffn_only] [--prefix N] [--warmups N] "
               "[--samples N] [--tokens N] [--capacity N] "
               "[--materialize-every-token 0|1] MODEL\n",
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
    } else if (std::strcmp(arg, "--materialize-every-token") == 0 &&
               index + 1 < argc) {
      options->materialize_every_token = std::strcmp(argv[++index], "1") == 0;
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

void apply_candidate(bool candidate, bool materialize_every_token) {
  qw38::cuda::set_lazy_output_materialization_override(
      candidate && !materialize_every_token);
  qw38::cuda::set_output_commit_overlap_override(candidate);
}

void clear_candidate() {
  qw38::cuda::clear_lazy_output_materialization_override();
  qw38::cuda::clear_output_commit_overlap_override();
}

qw38::Status run_token(const qw38::cuda::ResidentModel& model, std::size_t token,
                       qw38::cuda::SchedulerSession* session,
                       qw38::cuda::SchedulerWorkspace* workspace, float* logits,
                       float* hidden, float* elapsed,
                       qw38::cuda::SchedulerGraphs* graphs) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, nullptr, nullptr,
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
                           qw38::cuda::SchedulerWorkspace* workspace,
                           qw38::cuda::SchedulerGraphs* graphs) {
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphFfnOnly);
  return graphs->create(model, workspace);
}

void print_counts(std::size_t warmups, std::size_t samples,
                  std::size_t candidates, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-118\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

int run_inventory(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  workspace.reset_transfer_counters();
  clear_candidate();
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  const std::uint64_t prefill_d2h = workspace.transfer_d2h_bytes_;
  const std::uint64_t prefill_d2d = workspace.transfer_d2d_bytes_;
  workspace.reset_transfer_counters();
  float elapsed = 0.0F;
  if (status.is_ok()) {
    status = run_token(model, tokens[prefix], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs);
  }
  const std::uint64_t decode_d2h = workspace.transfer_d2h_bytes_;
  const std::uint64_t decode_d2d = workspace.transfer_d2d_bytes_;
  const std::uint32_t decode_d2h_copies = workspace.transfer_d2h_copies_;
  workspace.reset_transfer_counters();
  apply_candidate(true, false);
  if (status.is_ok()) {
    status = run_token(model, tokens[prefix + 1], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs);
  }
  const std::uint64_t lazy_d2h = workspace.transfer_d2h_bytes_;
  const std::uint64_t lazy_d2d = workspace.transfer_d2d_bytes_;
  clear_candidate();
  const bool ok = status.is_ok() && decode_d2h >= kLogitsBytes + kHiddenBytes &&
                  lazy_d2h < decode_d2h;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-118\",\"workload\":\"inventory\","
      "\"ok\":%s,\"parent\":\"post113_selected\",\"execution_graphs\":\"ffn_only\","
      "\"logits_bytes\":%zu,\"hidden_bytes\":%zu,"
      "\"gdn_carry_bytes\":%zu,\"gdn_carry_production_traffic\":false,"
      "\"prompt_chunk_rows\":%zu,\"prompt_microbatch_rows\":%zu,"
      "\"prefill_d2h_bytes\":%llu,\"prefill_d2d_bytes\":%llu,"
      "\"eager_decode_d2h_bytes\":%llu,\"eager_decode_d2d_bytes\":%llu,"
      "\"eager_decode_d2h_copies\":%u,\"lazy_decode_d2h_bytes\":%llu,"
      "\"lazy_decode_d2d_bytes\":%llu,\"changes_selected\":2,"
      "\"candidate_a\":\"lazy_logits_hidden_device_greedy\","
      "\"candidate_b\":\"decode_d2h_overlap_skip_duplicate_host_copy\","
      "\"opt110_hooks_ready\":%s,\"keep\":false}\n",
      kPrefix, json_bool(ok), kLogitsBytes, kHiddenBytes,
      static_cast<std::size_t>(48) *
          (qw38::internal::kGdnConvolutionValues +
           qw38::internal::kGdnRecurrentStateValues) *
          sizeof(float),
      qw38::cuda::kPromptChunkRows,
      qw38::cuda::selected_prompt_microbatch_rows(),
      static_cast<unsigned long long>(prefill_d2h),
      static_cast<unsigned long long>(prefill_d2d),
      static_cast<unsigned long long>(decode_d2h),
      static_cast<unsigned long long>(decode_d2d), decode_d2h_copies,
      static_cast<unsigned long long>(lazy_d2h),
      static_cast<unsigned long long>(lazy_d2d),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()));
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_greedy_parity(const qw38::cuda::ResidentModel& model,
                      const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession eager;
  qw38::cuda::SchedulerWorkspace eager_ws;
  qw38::cuda::SchedulerSession lazy;
  qw38::cuda::SchedulerWorkspace lazy_ws;
  qw38::cuda::SchedulerGraphs eager_graphs;
  qw38::cuda::SchedulerGraphs lazy_graphs;
  qw38::Status status = eager.create(capacity);
  if (status.is_ok()) status = eager_ws.create(capacity);
  if (status.is_ok()) status = lazy.create(capacity);
  if (status.is_ok()) status = lazy_ws.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &eager_ws, &eager_graphs);
  if (status.is_ok()) status = create_graphs(model, &lazy_ws, &lazy_graphs);
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::vector<float> lazy_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  std::array<float, qw38::internal::kResidualWidth> lazy_hidden{};
  clear_candidate();
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &eager, &eager_ws, &eager_graphs,
                     eager_logits.data(), eager_hidden.data());
  }
  apply_candidate(true, false);
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &lazy, &lazy_ws, &lazy_graphs,
                     lazy_logits.data(), lazy_hidden.data());
  }
  std::size_t eager_token = 0;
  std::size_t lazy_token = 0;
  if (status.is_ok()) status = qw38::cuda::greedy_sample(eager, &eager_token);
  if (status.is_ok()) status = qw38::cuda::greedy_sample(lazy, &lazy_token);
  if (status.is_ok()) {
    status = lazy.copy_last_outputs(lazy_logits.data(), lazy_logits.size(),
                                    lazy_hidden.data(), lazy_hidden.size());
  }
  const bool logits_exact =
      std::memcmp(eager_logits.data(), lazy_logits.data(),
                  eager_logits.size() * sizeof(float)) == 0;
  const bool hidden_exact =
      std::memcmp(eager_hidden.data(), lazy_hidden.data(),
                  eager_hidden.size() * sizeof(float)) == 0;
  clear_candidate();
  const bool ok = status.is_ok() && eager_token == lazy_token && logits_exact &&
                  hidden_exact && lazy.greedy_token_valid();
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-118\","
      "\"workload\":\"greedy-parity\",\"ok\":%s,\"exact\":%s,"
      "\"greedy_equal\":%s,\"eager_token\":%zu,\"lazy_token\":%zu,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(logits_exact && hidden_exact),
      json_bool(eager_token == lazy_token), eager_token, lazy_token);
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_api(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 3, options.capacity);
  std::vector<std::size_t> tokens(prefix + 3);
  fill_tokens(&tokens);
  apply_candidate(true, false);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
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
    status = run_token(model, tokens[prefix], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs);
  }
  std::vector<float> after_eval(qw38::internal::kVocabularySize);
  if (status.is_ok()) {
    status = session.copy_last_outputs(after_eval.data(), after_eval.size(),
                                       hidden.data(), hidden.size());
  }
  const bool eval_changed =
      std::memcmp(before.data(), after_eval.data(),
                  before.size() * sizeof(float)) != 0;
  const char* tmp = "/tmp/qw38-opt118-ckpt.bin";
  if (status.is_ok()) status = session.save_checkpoint(tmp);
  qw38::cuda::SchedulerSession restored;
  if (status.is_ok()) status = restored.create(capacity);
  if (status.is_ok()) status = restored.restore_checkpoint(tmp, &workspace);
  bool equal = false;
  if (status.is_ok()) status = restored.state_equals(session, &equal);
  std::remove(tmp);
  PollContext poll{0, 1};
  const qw38::cuda::EvalControl cancel{
      [](void* context) noexcept -> qw38::Status {
        auto* ctx = static_cast<PollContext*>(context);
        ++ctx->calls;
        if (ctx->stop_after != 0 && ctx->calls >= ctx->stop_after) {
          return {qw38::StatusCode::kCancelled, "cancelled"};
        }
        return qw38::Status::ok();
      },
      &poll};
  const std::size_t frontier_before = session.frontier();
  qw38::Status cancelled = qw38::cuda::execute_token(
      model, tokens[prefix + 1], &session, &workspace, logits.data(),
      qw38::internal::kVocabularySize, hidden.data(),
      qw38::internal::kResidualWidth, &elapsed, &cancel, nullptr,
      qw38::cuda::PointwisePath::kFused, &graphs, nullptr);
  const bool cancel_isolated = !cancelled.is_ok() &&
                               session.frontier() == frontier_before;
  clear_candidate();
  const bool ok = status.is_ok() && repeat_equal && eval_changed && equal &&
                  cancel_isolated;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-118\",\"workload\":\"api\","
      "\"ok\":%s,\"repeat_logits_equal\":%s,\"eval_changes_logits\":%s,"
      "\"restore_equals\":%s,\"cancel_isolated\":%s,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(repeat_equal), json_bool(eval_changed),
      json_bool(equal), json_bool(cancel_isolated));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_cancellation(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  return run_api(model, options);
}

int run_state_memory(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  (void)model;
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 1, options.capacity);
  apply_candidate(true, false);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  const std::size_t extra = session.allocated_bytes();
  clear_candidate();
  const bool ok = status.is_ok() && extra > kLogitsBytes;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-118\","
      "\"workload\":\"state-memory\",\"ok\":%s,\"session_bytes\":%zu,"
      "\"extra_device_output_bytes\":%zu,\"keep\":false}\n",
      kPrefix, json_bool(ok), extra, kLogitsBytes + kHiddenBytes + sizeof(int));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

qw38::Status run_session_arm(const qw38::cuda::ResidentModel& model,
                             const std::vector<std::size_t>& tokens,
                             std::size_t prefix, std::size_t outputs,
                             bool candidate, bool materialize_every_token,
                             bool prefill_only, ArmResult* out) {
  apply_candidate(candidate, materialize_every_token);
  const std::size_t capacity = session_capacity(prefix, outputs, 0);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  workspace.reset_transfer_counters();
  const auto started = std::chrono::steady_clock::now();
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  const float ttft = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count());
  std::vector<float> latencies;
  latencies.reserve(outputs);
  if (!prefill_only) {
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float elapsed = 0.0F;
      const auto token_started = std::chrono::steady_clock::now();
      status = run_token(model, tokens[prefix + step], &session, &workspace,
                         logits.data(), hidden.data(), &elapsed, &graphs);
      if (status.is_ok() && materialize_every_token) {
        status = session.copy_last_outputs(logits.data(), logits.size(),
                                           hidden.data(), hidden.size());
      }
      if (status.is_ok() && !materialize_every_token && candidate) {
        std::size_t sampled = 0;
        status = qw38::cuda::greedy_sample(session, &sampled);
        (void)sampled;
      }
      latencies.push_back(static_cast<float>(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - token_started)
              .count()));
    }
  }
  out->wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count());
  const std::size_t counted = prefill_only ? prefix : outputs;
  out->tok_s = out->wall_ms > 0.0F
                   ? static_cast<float>(counted) * 1000.0F / out->wall_ms
                   : 0.0F;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  out->ttft_ms = ttft;
  out->d2h_bytes = workspace.transfer_d2h_bytes_;
  out->d2d_bytes = workspace.transfer_d2d_bytes_;
  out->d2h_copies = workspace.transfer_d2h_copies_;
  clear_candidate();
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"wall_ms\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,"
      "\"p95_ms\":%.9g,\"ttft_ms\":%.9g,\"d2h_bytes\":%llu,"
      "\"d2d_bytes\":%llu,\"d2h_copies\":%u}%s",
      name, static_cast<double>(arm.wall_ms), static_cast<double>(arm.tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms),
      static_cast<double>(arm.ttft_ms),
      static_cast<unsigned long long>(arm.d2h_bytes),
      static_cast<unsigned long long>(arm.d2d_bytes), arm.d2h_copies,
      last ? "" : ",");
}

int run_session_ab(const qw38::cuda::ResidentModel& model,
                   const Options& options) {
  const bool prefill_only = options.prefix == 4096 && options.tokens == 0;
  const std::size_t prompt = prefill_only ? 4096 : options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_session_arm(model, tokens, prompt, outputs, false,
                             options.materialize_every_token, prefill_only,
                             &discarded);
    if (status.is_ok()) {
      status = run_session_arm(model, tokens, prompt, outputs, true,
                               options.materialize_every_token, prefill_only,
                               &discarded);
    }
    if (!status.is_ok()) return fail_status(status);
    std::printf(
        "quartz_warmup=%zu tok_s=%.9g wall_ms=%.9g independently_restored=true\n",
        warmup, static_cast<double>(discarded.tok_s),
        static_cast<double>(discarded.wall_ms));
  }
  std::vector<ArmResult> arm_a(options.samples);
  std::vector<ArmResult> arm_b(options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    const bool ba = (pair % 2) == 1;
    ArmResult first;
    ArmResult second;
    status = run_session_arm(model, tokens, prompt, outputs, ba,
                             options.materialize_every_token, prefill_only,
                             &first);
    if (status.is_ok()) {
      status = run_session_arm(model, tokens, prompt, outputs, !ba,
                               options.materialize_every_token, prefill_only,
                               &second);
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
      "%s{\"schema_version\":1,\"task\":\"OPT-118\",\"workload\":\"session-ab\","
      "\"prefix\":%zu,\"prefill\":%s,\"materialize_every_token\":%s,"
      "\"warmups\":%zu,\"samples\":%zu,\"tokens\":%zu,"
      "\"parent\":\"eager_blocking\",\"candidate\":\"lazy_overlap\","
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"acceptance\",\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true,\"pairs\":[",
      kPrefix, prompt, json_bool(prefill_only),
      json_bool(options.materialize_every_token), options.warmups,
      options.samples, outputs, options.warmups, options.samples);
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

}  // namespace

int main(int argc, char** argv) {
  Options options;
  const int model_index = parse_args(argc, argv, &options);
  if (model_index < 0) return model_index == -1 ? 1 : model_index;
  qw38::internal::MappedFile mapping;
  qw38::cuda::ResidentModel model;
  const qw38::Status status = load_model(argv[model_index], &mapping, &model);
  if (!status.is_ok()) return fail_status(status);
  if (std::strcmp(options.workload, "inventory") == 0) {
    return run_inventory(model, options);
  }
  if (std::strcmp(options.workload, "greedy-parity") == 0) {
    return run_greedy_parity(model, options);
  }
  if (std::strcmp(options.workload, "api") == 0) {
    return run_api(model, options);
  }
  if (std::strcmp(options.workload, "cancellation") == 0) {
    return run_cancellation(model, options);
  }
  if (std::strcmp(options.workload, "state-memory") == 0) {
    return run_state_memory(model, options);
  }
  if (std::strcmp(options.workload, "session-ab") == 0) {
    return run_session_ab(model, options);
  }
  return usage(argv[0]);
}
