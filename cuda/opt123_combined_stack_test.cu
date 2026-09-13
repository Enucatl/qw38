#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "opt110_engine_hook.cuh"
#include "opt120_packed_kv.cuh"
#include "opt121_weight_requant.cuh"
#include "opt122_gdn_state_precision.cuh"
#include "q4k_decode_path.cuh"
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

constexpr char kPrefix[] = "QW38_OPT123_COMBINED_STACK_RESULT=";
constexpr char kCounts[] = "QW38_OPT123_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 8192;
constexpr std::size_t kCapacity128k = 131072;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;
constexpr std::size_t kLayers = 64;

enum class Stack : int { kControl = 0, kOpt118 = 1, kOpt119 = 2, kCombined = 3 };

struct Options final {
  const char* workload = "inventory";
  const char* execution_graphs = "ffn_only";
  const char* stack_a = "control";
  const char* stack_b = "combined";
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
  float ttft_ms = 0.0F;
  std::size_t session_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::uint64_t d2h_bytes = 0;
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
               "state-memory|session-ab|long-context] "
               "[--execution-graphs ffn_only] "
               "[--stack-a control|opt118|opt119|combined] "
               "[--stack-b control|opt118|opt119|combined] "
               "[--prefix N] [--warmups N] [--samples N] [--tokens N] "
               "[--capacity N] MODEL\n",
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

bool parse_stack(const char* text, Stack* stack) {
  if (std::strcmp(text, "control") == 0) {
    *stack = Stack::kControl;
    return true;
  }
  if (std::strcmp(text, "opt118") == 0) {
    *stack = Stack::kOpt118;
    return true;
  }
  if (std::strcmp(text, "opt119") == 0) {
    *stack = Stack::kOpt119;
    return true;
  }
  if (std::strcmp(text, "combined") == 0) {
    *stack = Stack::kCombined;
    return true;
  }
  return false;
}

const char* stack_name(Stack stack) {
  switch (stack) {
    case Stack::kControl:
      return "control";
    case Stack::kOpt118:
      return "opt118";
    case Stack::kOpt119:
      return "opt119";
    case Stack::kCombined:
      return "combined";
  }
  return "unknown";
}

int parse_args(int argc, char** argv, Options* options) {
  int model_index = -1;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--execution-graphs") == 0 && index + 1 < argc) {
      options->execution_graphs = argv[++index];
    } else if (std::strcmp(arg, "--stack-a") == 0 && index + 1 < argc) {
      options->stack_a = argv[++index];
    } else if (std::strcmp(arg, "--stack-b") == 0 && index + 1 < argc) {
      options->stack_b = argv[++index];
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
  if (std::strcmp(options->execution_graphs,
                  qw38::cuda::kLegalExecutionGraphFfnOnly) != 0) {
    std::fprintf(stderr, "OPT-123 retains ffn_only; rejected OPT-117 leak\n");
    return 2;
  }
  Stack unused;
  if (!parse_stack(options->stack_a, &unused) ||
      !parse_stack(options->stack_b, &unused)) {
    std::fprintf(stderr, "invalid stack ident\n");
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

void apply_stack(Stack stack) {
  const bool lazy = stack == Stack::kOpt118 || stack == Stack::kCombined;
  const bool fusion = stack == Stack::kOpt119 || stack == Stack::kCombined;
  qw38::cuda::set_lazy_output_materialization_override(lazy);
  qw38::cuda::set_output_commit_overlap_override(lazy);
  qw38::cuda::set_mixer_norm_q8_fusion_override(fusion);
  qw38::cuda::set_ffn_norm_q8_fusion_override(fusion);
}

void clear_stack() {
  qw38::cuda::clear_lazy_output_materialization_override();
  qw38::cuda::clear_output_commit_overlap_override();
  qw38::cuda::clear_mixer_norm_q8_fusion_override();
  qw38::cuda::clear_ffn_norm_q8_fusion_override();
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
      "%s{\"schema_version\":1,\"task\":\"OPT-123\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

qw38::Status materialize(qw38::cuda::SchedulerSession* session, float* logits,
                         float* hidden) {
  return session->copy_last_outputs(logits, qw38::internal::kVocabularySize,
                                    hidden, qw38::internal::kResidualWidth);
}

int run_inventory(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  const Stack stacks[] = {Stack::kControl, Stack::kOpt118, Stack::kOpt119,
                          Stack::kCombined};
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-123\",\"workload\":\"inventory\","
      "\"ok\":true,\"parent\":\"post113_selected\","
      "\"execution_graphs\":\"ffn_only\","
      "\"q4_decode\":\"%s\",\"packed_kv\":\"%s\",\"weight_requant\":\"%s\","
      "\"gdn_state\":\"%s\",\"opt117_rejected_ffn_only\":true,"
      "\"opt120_rejected_dense_kv\":true,\"opt121_rejected_none_requant\":true,"
      "\"opt122_rejected_fp32_gdn\":true,\"opt110_hooks_ready\":%s,"
      "\"prefix\":%zu,\"layers\":%zu,\"stacks\":[",
      kPrefix, qw38::cuda::kSelectedQ4DecodePath,
      qw38::cuda::packed_kv_format_ident(qw38::cuda::kSelectedPackedKvFormat),
      qw38::cuda::kSelectedWeightRequantConfig,
      qw38::cuda::gdn_state_format_ident(qw38::cuda::kSelectedGdnStateFormat),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()), prefix, kLayers);
  bool all_ok = true;
  qw38::Status status = qw38::Status::ok();
  for (int index = 0; index < 4; ++index) {
    apply_stack(stacks[index]);
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) status = create_graphs(model, &workspace, &graphs);
    workspace.reset_transfer_counters();
    if (status.is_ok()) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    const std::uint64_t prefill_d2h = workspace.transfer_d2h_bytes_;
    const std::uint32_t mixer_q = workspace.activation_mixer_quantize_launches_;
    const std::uint32_t ffn_q = workspace.activation_ffn_quantize_launches_;
    const std::uint32_t mixer_f = workspace.activation_mixer_norm_q8_launches_;
    const std::uint32_t ffn_f = workspace.activation_ffn_norm_q8_launches_;
    workspace.reset_transfer_counters();
    float elapsed = 0.0F;
    if (status.is_ok()) {
      status = run_token(model, tokens[prefix], &session, &workspace,
                         logits.data(), hidden.data(), &elapsed, &graphs);
    }
    const std::uint64_t decode_d2h = workspace.transfer_d2h_bytes_;
    clear_stack();
    const bool fusion = stacks[index] == Stack::kOpt119 ||
                        stacks[index] == Stack::kCombined;
    const bool lazy = stacks[index] == Stack::kOpt118 ||
                      stacks[index] == Stack::kCombined;
    const bool launches_ok =
        fusion ? (mixer_q == 0 && ffn_q == 0 && mixer_f > 0 && ffn_f > 0)
               : (mixer_q > 0 && ffn_q > 0);
    const bool stack_ok = status.is_ok() && launches_ok;
    all_ok = all_ok && stack_ok;
    if (index != 0) std::printf(",");
    std::printf(
        "{\"id\":\"%s\",\"ok\":%s,\"lazy\":%s,\"fusion\":%s,"
        "\"prefill_d2h_bytes\":%llu,\"decode_d2h_bytes\":%llu,"
        "\"mixer_quantize_launches\":%u,\"ffn_quantize_launches\":%u,"
        "\"mixer_norm_q8_launches\":%u,\"ffn_norm_q8_launches\":%u,"
        "\"session_bytes\":%zu,\"workspace_bytes\":%zu}",
        stack_name(stacks[index]), json_bool(stack_ok), json_bool(lazy),
        json_bool(fusion), static_cast<unsigned long long>(prefill_d2h),
        static_cast<unsigned long long>(decode_d2h), mixer_q, ffn_q, mixer_f,
        ffn_f, session.allocated_bytes(), workspace.allocated_bytes());
  }
  const bool no_leak =
      std::strcmp(qw38::cuda::kSelectedExecutionGraphPath, "ffn_only") == 0 &&
      qw38::cuda::kSelectedPackedKvFormat ==
          qw38::cuda::PackedKvFormat::kDenseBf16 &&
      std::strcmp(qw38::cuda::kSelectedWeightRequantConfig, "none") == 0 &&
      qw38::cuda::kSelectedGdnStateFormat == qw38::cuda::GdnStateFormat::kFp32;
  std::printf("],\"rejected_no_leak\":%s,\"ok\":%s,\"keep\":false}\n",
              json_bool(no_leak), json_bool(all_ok && no_leak));
  print_counts(0, 1, 4, false);
  return all_ok && no_leak ? 0 : 1;
}

int run_greedy_parity(const qw38::cuda::ResidentModel& model,
                      const Options& options) {
  Stack stack_a;
  Stack stack_b;
  parse_stack(options.stack_a, &stack_a);
  parse_stack(options.stack_b, &stack_b);
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession parent;
  qw38::cuda::SchedulerWorkspace parent_ws;
  qw38::cuda::SchedulerSession cand;
  qw38::cuda::SchedulerWorkspace cand_ws;
  qw38::cuda::SchedulerGraphs parent_graphs;
  qw38::cuda::SchedulerGraphs cand_graphs;
  qw38::Status status = parent.create(capacity);
  if (status.is_ok()) status = parent_ws.create(capacity);
  if (status.is_ok()) status = cand.create(capacity);
  if (status.is_ok()) status = cand_ws.create(capacity);
  apply_stack(stack_a);
  if (status.is_ok()) status = create_graphs(model, &parent_ws, &parent_graphs);
  apply_stack(stack_b);
  if (status.is_ok()) status = create_graphs(model, &cand_ws, &cand_graphs);
  std::vector<float> parent_logits(qw38::internal::kVocabularySize);
  std::vector<float> cand_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> parent_hidden{};
  std::array<float, qw38::internal::kResidualWidth> cand_hidden{};
  apply_stack(stack_a);
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &parent, &parent_ws, &parent_graphs,
                     parent_logits.data(), parent_hidden.data());
  }
  if (status.is_ok()) {
    status = materialize(&parent, parent_logits.data(), parent_hidden.data());
  }
  apply_stack(stack_b);
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &cand, &cand_ws, &cand_graphs,
                     cand_logits.data(), cand_hidden.data());
  }
  if (status.is_ok()) {
    status = materialize(&cand, cand_logits.data(), cand_hidden.data());
  }
  std::size_t parent_token = 0;
  std::size_t cand_token = 0;
  apply_stack(stack_a);
  if (status.is_ok()) status = qw38::cuda::greedy_sample(parent, &parent_token);
  apply_stack(stack_b);
  if (status.is_ok()) status = qw38::cuda::greedy_sample(cand, &cand_token);
  const bool logits_exact =
      std::memcmp(parent_logits.data(), cand_logits.data(),
                  parent_logits.size() * sizeof(float)) == 0;
  const bool hidden_exact =
      std::memcmp(parent_hidden.data(), cand_hidden.data(),
                  parent_hidden.size() * sizeof(float)) == 0;
  clear_stack();
  const bool ok = status.is_ok() && parent_token == cand_token && logits_exact &&
                  hidden_exact;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-123\","
      "\"workload\":\"greedy-parity\",\"ok\":%s,\"exact\":%s,"
      "\"greedy_equal\":%s,\"parent_token\":%zu,\"candidate_token\":%zu,"
      "\"stack_a\":\"%s\",\"stack_b\":\"%s\",\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(logits_exact && hidden_exact),
      json_bool(parent_token == cand_token), parent_token, cand_token,
      stack_name(stack_a), stack_name(stack_b));
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_api(const qw38::cuda::ResidentModel& model, const Options& options) {
  Stack stack;
  parse_stack(options.stack_b, &stack);
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 8, options.capacity);
  std::vector<std::size_t> tokens(prefix + 8);
  fill_tokens(&tokens);
  apply_stack(stack);
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
  const char* tmp = "/tmp/qw38-opt123-ckpt.bin";
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
  const bool cancel_isolated =
      !cancelled.is_ok() && session.frontier() == frontier_before;
  PollContext live_poll{0, 0};
  const qw38::cuda::EvalControl live{
      [](void* context) noexcept -> qw38::Status {
        auto* ctx = static_cast<PollContext*>(context);
        ++ctx->calls;
        return qw38::Status::ok();
      },
      &live_poll};
  const auto request_started = std::chrono::steady_clock::now();
  float ttft_ms = 0.0F;
  std::vector<float> latencies;
  latencies.reserve(8);
  std::size_t emitted = 0;
  for (std::size_t step = 0; status.is_ok() && step < 8; ++step) {
    const auto token_started = std::chrono::steady_clock::now();
    status = qw38::cuda::execute_token(
        model, tokens[prefix + step], &session, &workspace, logits.data(),
        qw38::internal::kVocabularySize, hidden.data(),
        qw38::internal::kResidualWidth, &elapsed, &live, nullptr,
        qw38::cuda::PointwisePath::kFused, &graphs, nullptr);
    const float token_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - token_started)
            .count());
    latencies.push_back(token_ms);
    if (step == 0) {
      ttft_ms = static_cast<float>(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - request_started)
              .count());
    }
    ++emitted;
  }
  const float wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - request_started)
          .count());
  const std::size_t peak = session.allocated_bytes() + workspace.allocated_bytes();
  clear_stack();
  const bool ok = status.is_ok() && repeat_equal && eval_changed && equal &&
                  cancel_isolated && live_poll.calls > 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-123\",\"workload\":\"api\","
      "\"ok\":%s,\"repeat_logits_equal\":%s,\"eval_changes_logits\":%s,"
      "\"restore_equals\":%s,\"cancel_isolated\":%s,\"stack\":\"%s\","
      "\"request_loop\":true,\"ttft_ms\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,"
      "\"wall_ms\":%.9g,\"emitted_tokens\":%zu,\"peak_memory_bytes\":%zu,"
      "\"cancellation_callbacks\":%zu,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(repeat_equal), json_bool(eval_changed),
      json_bool(equal), json_bool(cancel_isolated), stack_name(stack),
      static_cast<double>(ttft_ms),
      static_cast<double>(percentile(latencies, 0.50F)),
      static_cast<double>(percentile(latencies, 0.95F)),
      static_cast<double>(wall_ms), emitted, peak, live_poll.calls);
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_state_memory(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  (void)model;
  Stack stack;
  parse_stack(options.stack_b, &stack);
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 1, options.capacity);
  apply_stack(stack);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  const std::size_t extra = session.allocated_bytes();
  const std::size_t work = workspace.allocated_bytes();
  clear_stack();
  const bool ok = status.is_ok() && extra > 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-123\","
      "\"workload\":\"state-memory\",\"ok\":%s,\"session_bytes\":%zu,"
      "\"workspace_bytes\":%zu,\"stack\":\"%s\",\"keep\":false}\n",
      kPrefix, json_bool(ok), extra, work, stack_name(stack));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

qw38::Status run_session_arm(const qw38::cuda::ResidentModel& model,
                             const std::vector<std::size_t>& tokens,
                             std::size_t prefix, std::size_t outputs,
                             Stack stack, bool prefill_only,
                             std::size_t requested_capacity, ArmResult* out) {
  apply_stack(stack);
  const std::size_t capacity =
      session_capacity(prefix, outputs, requested_capacity);
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
  out->session_bytes = session.allocated_bytes();
  out->workspace_bytes = workspace.allocated_bytes();
  out->d2h_bytes = workspace.transfer_d2h_bytes_;
  clear_stack();
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"wall_ms\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,"
      "\"p95_ms\":%.9g,\"ttft_ms\":%.9g,\"session_bytes\":%zu,"
      "\"workspace_bytes\":%zu,\"d2h_bytes\":%llu}%s",
      name, static_cast<double>(arm.wall_ms), static_cast<double>(arm.tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms),
      static_cast<double>(arm.ttft_ms), arm.session_bytes, arm.workspace_bytes,
      static_cast<unsigned long long>(arm.d2h_bytes), last ? "" : ",");
}

int run_session_ab(const qw38::cuda::ResidentModel& model,
                   const Options& options) {
  Stack stack_a;
  Stack stack_b;
  parse_stack(options.stack_a, &stack_a);
  parse_stack(options.stack_b, &stack_b);
  const bool prefill_only = options.tokens == 0;
  const std::size_t prompt = options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_session_arm(model, tokens, prompt, outputs, stack_a,
                             prefill_only, 0, &discarded);
    if (status.is_ok()) {
      status = run_session_arm(model, tokens, prompt, outputs, stack_b,
                               prefill_only, 0, &discarded);
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
    status = run_session_arm(model, tokens, prompt, outputs,
                             ba ? stack_b : stack_a, prefill_only, 0, &first);
    if (status.is_ok()) {
      status = run_session_arm(model, tokens, prompt, outputs,
                               ba ? stack_a : stack_b, prefill_only, 0, &second);
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
      "%s{\"schema_version\":1,\"task\":\"OPT-123\",\"workload\":\"session-ab\","
      "\"prefix\":%zu,\"prefill\":%s,"
      "\"warmups\":%zu,\"samples\":%zu,\"tokens\":%zu,"
      "\"stack_a\":\"%s\",\"stack_b\":\"%s\","
      "\"parent\":\"post113_selected\",\"candidate\":\"combined_opt118_opt119\","
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"acceptance\",\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true,\"pairs\":[",
      kPrefix, prompt, json_bool(prefill_only), options.warmups, options.samples,
      outputs, stack_name(stack_a), stack_name(stack_b), options.warmups,
      options.samples);
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

int run_long_context(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  Stack stack;
  parse_stack(options.stack_b, &stack);
  const std::size_t prefix = options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 32 : options.tokens;
  const std::size_t capacity =
      options.capacity == 0 ? kCapacity128k : options.capacity;
  if (prefix + outputs > capacity) {
    std::fprintf(stderr, "prefix+outputs %zu exceeds capacity %zu\n",
                 prefix + outputs, capacity);
    return 2;
  }
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  ArmResult measured;
  const qw38::Status status = run_session_arm(
      model, tokens, prefix, outputs, stack, false, capacity, &measured);
  if (!status.is_ok()) return fail_status(status);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-123\","
      "\"workload\":\"long-context\",\"ok\":true,\"prefix\":%zu,\"tokens\":%zu,"
      "\"capacity\":%zu,\"stack\":\"%s\",\"wall_ms\":%.9g,\"ttft_ms\":%.9g,"
      "\"tok_s\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,"
      "\"session_bytes\":%zu,\"workspace_bytes\":%zu,"
      "\"populated_cache\":true,\"capacity_alone_is_not_traffic\":true,"
      "\"keep\":false}\n",
      kPrefix, prefix, outputs, capacity, stack_name(stack),
      static_cast<double>(measured.wall_ms),
      static_cast<double>(measured.ttft_ms), static_cast<double>(measured.tok_s),
      static_cast<double>(measured.p50_ms), static_cast<double>(measured.p95_ms),
      measured.session_bytes, measured.workspace_bytes);
  print_counts(0, 1, 1, false);
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
  if (std::strcmp(options.workload, "state-memory") == 0) {
    return run_state_memory(model, options);
  }
  if (std::strcmp(options.workload, "session-ab") == 0) {
    return run_session_ab(model, options);
  }
  if (std::strcmp(options.workload, "long-context") == 0) {
    return run_long_context(model, options);
  }
  return usage(argv[0]);
}
