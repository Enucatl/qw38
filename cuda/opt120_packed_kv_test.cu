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

constexpr char kPrefix[] = "QW38_OPT120_PACKED_KV_RESULT=";
constexpr char kCounts[] = "QW38_OPT120_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 8192;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;
constexpr std::size_t kTailPrefixes[] = {2048, 5120, 8192};

struct Options final {
  const char* workload = "inventory";
  const char* packed_kv = "q8q8";
  const char* execution_graphs = "ffn_only";
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
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload reference|inventory|greedy-parity|api|"
               "format|state-memory|session-ab] "
               "[--packed-kv q8q8|q8q4|q4q4] "
               "[--execution-graphs ffn_only] [--prefix N] [--warmups N] "
               "[--samples N] [--tokens N] [--capacity N] MODEL\n",
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
    } else if (std::strcmp(arg, "--packed-kv") == 0 && index + 1 < argc) {
      options->packed_kv = argv[++index];
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

void apply_candidate(bool candidate, const char* packed_kv) {
  if (candidate) {
    qw38::cuda::set_packed_kv_format_override(
        qw38::cuda::packed_kv_format_from_ident(packed_kv));
  } else {
    qw38::cuda::set_packed_kv_format_override(qw38::cuda::PackedKvFormat::kDenseBf16);
  }
}

void clear_candidate() {
  qw38::cuda::clear_packed_kv_format_override();
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
      "%s{\"schema_version\":1,\"task\":\"OPT-120\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

int run_reference(const Options&) {
  bool ok = true;
  const float zeros[32] = {};
  float outlier[32];
  float ramp[17];
  for (int i = 0; i < 32; ++i) outlier[i] = (i == 7) ? 12.5F : 0.1F;
  for (int i = 0; i < 17; ++i) ramp[i] = static_cast<float>(i) - 8.0F;
  std::int8_t codes[32];
  float scale = 0.0F;
  float back[32];
  qw38::cuda::packed_kv_quantize_group(zeros, 32, qw38::cuda::PackedKvTensor::kQ8,
                                       codes, &scale);
  qw38::cuda::packed_kv_dequantize_group(codes, 32, scale, back);
  for (int i = 0; i < 32; ++i) ok = ok && back[i] == 0.0F && codes[i] == 0;
  qw38::cuda::packed_kv_quantize_group(outlier, 32, qw38::cuda::PackedKvTensor::kQ8,
                                       codes, &scale);
  ok = ok && scale > 0.0F;
  qw38::cuda::packed_kv_quantize_group(ramp, 17, qw38::cuda::PackedKvTensor::kQ4,
                                       codes, &scale);
  std::uint8_t packed[16];
  qw38::cuda::packed_kv_pack_q4(codes, 17, packed);
  std::int8_t unpacked[17];
  qw38::cuda::packed_kv_unpack_q4(packed, 17, unpacked);
  for (int i = 0; i < 17; ++i) ok = ok && unpacked[i] == codes[i];
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-120\",\"workload\":\"reference\","
      "\"ok\":%s,\"zeros\":true,\"outliers\":true,\"incomplete_block\":17,"
      "\"group\":32,\"q8_clip\":127,\"q4_clip\":7,\"crossover\":1024,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_format(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 1, options.capacity);
  std::vector<std::size_t> tokens(prefix);
  fill_tokens(&tokens);
  apply_candidate(true, options.packed_kv);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &session, &workspace, nullptr,
                     logits.data(), hidden.data());
  }
  const char* packed_path = "/tmp/qw38-opt120-packed.ckpt";
  const char* dense_path = "/tmp/qw38-opt120-dense.ckpt";
  if (status.is_ok()) status = session.save_checkpoint(packed_path);
  bool packed_restore = false;
  if (status.is_ok()) {
    const qw38::Status restored =
        session.restore_checkpoint(packed_path, &workspace);
    packed_restore = restored.is_ok();
    if (!restored.is_ok()) status = restored;
  }
  clear_candidate();
  qw38::cuda::SchedulerSession dense;
  qw38::cuda::SchedulerWorkspace dense_ws;
  qw38::Status dense_status = dense.create(capacity);
  if (dense_status.is_ok()) dense_status = dense_ws.create(capacity);
  bool legacy_rejected = false;
  if (dense_status.is_ok()) {
    const qw38::Status rejected =
        dense.restore_checkpoint(packed_path, &dense_ws);
    legacy_rejected = !rejected.is_ok();
  }
  bool dense_saved = false;
  if (dense_status.is_ok()) {
    dense_saved = dense.save_checkpoint(dense_path).is_ok();
  }
  apply_candidate(true, options.packed_kv);
  bool dense_into_packed_rejected = false;
  if (dense_saved) {
    const qw38::Status rejected =
        session.restore_checkpoint(dense_path, &workspace);
    dense_into_packed_rejected = !rejected.is_ok();
  }
  clear_candidate();
  const bool ok = status.is_ok() && packed_restore && legacy_rejected &&
                  dense_into_packed_rejected;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-120\",\"workload\":\"format\","
      "\"ok\":%s,\"packed_restore\":%s,\"packed_into_dense_rejected\":%s,"
      "\"dense_into_packed_rejected\":%s,\"versioned\":true,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(packed_restore),
      json_bool(legacy_rejected), json_bool(dense_into_packed_rejected));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_inventory(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  (void)model;
  const std::size_t capacity =
      options.capacity == 0 ? 131072 : options.capacity;
  const auto fmt =
      qw38::cuda::packed_kv_format_from_ident(options.packed_kv);
  const std::uint64_t dense = qw38::cuda::packed_kv_cache_bytes(
      qw38::cuda::PackedKvFormat::kDenseBf16, 16, capacity);
  const std::uint64_t q8q8 = qw38::cuda::packed_kv_cache_bytes(
      qw38::cuda::PackedKvFormat::kQ8Q8, 16, capacity);
  const std::uint64_t q8q4 = qw38::cuda::packed_kv_cache_bytes(
      qw38::cuda::PackedKvFormat::kQ8Q4, 16, capacity);
  const std::uint64_t q4q4 = qw38::cuda::packed_kv_cache_bytes(
      qw38::cuda::PackedKvFormat::kQ4Q4, 16, capacity);
  const std::uint64_t candidate =
      qw38::cuda::packed_kv_cache_bytes(fmt, 16, capacity);
  const bool ok = candidate < dense && q8q8 < dense;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-120\",\"workload\":\"inventory\","
      "\"ok\":%s,\"parent\":\"post113_selected\",\"execution_graphs\":\"ffn_only\","
      "\"primary_target\":\"long_context_decode\",\"integer_8bit\":true,"
      "\"fp8_rejected\":\"ds4_writes_floats\",\"group\":32,"
      "\"recent_window\":0,\"candidate\":\"%s\","
      "\"capacity\":%zu,\"dense_bytes\":%llu,\"q8q8_bytes\":%llu,"
      "\"q8q4_bytes\":%llu,\"q4q4_bytes\":%llu,\"candidate_bytes\":%llu,"
      "\"bytes_saved\":%llu,\"scales_included\":true,"
      "\"no_dense_shadow\":true,\"keep\":false}\n",
      kPrefix, json_bool(ok), options.packed_kv, capacity,
      static_cast<unsigned long long>(dense),
      static_cast<unsigned long long>(q8q8),
      static_cast<unsigned long long>(q8q4),
      static_cast<unsigned long long>(q4q4),
      static_cast<unsigned long long>(candidate),
      static_cast<unsigned long long>(dense - candidate));
  print_counts(0, 1, 3, false);
  return ok ? 0 : 1;
}

int run_greedy_parity(const qw38::cuda::ResidentModel& model,
                      const Options& options) {
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
  qw38::Status status = qw38::Status::ok();
  clear_candidate();
  status = parent.create(capacity);
  if (status.is_ok()) status = parent_ws.create(capacity);
  apply_candidate(true, options.packed_kv);
  if (status.is_ok()) status = cand.create(capacity);
  if (status.is_ok()) status = cand_ws.create(capacity);
  clear_candidate();
  if (status.is_ok()) status = create_graphs(model, &parent_ws, &parent_graphs);
  apply_candidate(true, options.packed_kv);
  if (status.is_ok()) status = create_graphs(model, &cand_ws, &cand_graphs);
  std::vector<float> parent_logits(qw38::internal::kVocabularySize);
  std::vector<float> cand_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> parent_hidden{};
  std::array<float, qw38::internal::kResidualWidth> cand_hidden{};
  clear_candidate();
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &parent, &parent_ws, &parent_graphs,
                     parent_logits.data(), parent_hidden.data());
  }
  apply_candidate(true, options.packed_kv);
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &cand, &cand_ws, &cand_graphs,
                     cand_logits.data(), cand_hidden.data());
  }
  std::size_t parent_token = 0;
  std::size_t cand_token = 0;
  if (status.is_ok()) status = qw38::cuda::greedy_sample(parent, &parent_token);
  if (status.is_ok()) status = qw38::cuda::greedy_sample(cand, &cand_token);
  const bool logits_exact =
      std::memcmp(parent_logits.data(), cand_logits.data(),
                  parent_logits.size() * sizeof(float)) == 0;
  const bool hidden_exact =
      std::memcmp(parent_hidden.data(), cand_hidden.data(),
                  parent_hidden.size() * sizeof(float)) == 0;
  clear_candidate();
  const bool ok = status.is_ok();
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-120\","
      "\"workload\":\"greedy-parity\",\"ok\":%s,\"exact\":%s,"
      "\"greedy_equal\":%s,\"parent_token\":%zu,\"candidate_token\":%zu,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(logits_exact && hidden_exact),
      json_bool(parent_token == cand_token), parent_token, cand_token);
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_api(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 3, options.capacity);
  std::vector<std::size_t> tokens(prefix + 3);
  fill_tokens(&tokens);
  apply_candidate(true, options.packed_kv);
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
  const char* tmp = "/tmp/qw38-opt120-ckpt.bin";
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
      "%s{\"schema_version\":1,\"task\":\"OPT-120\",\"workload\":\"api\","
      "\"ok\":%s,\"repeat_logits_equal\":%s,\"eval_changes_logits\":%s,"
      "\"restore_equals\":%s,\"cancel_isolated\":%s,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(repeat_equal), json_bool(eval_changed),
      json_bool(equal), json_bool(cancel_isolated));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_tails(const qw38::cuda::ResidentModel& model, const Options& options) {
  (void)options;
  bool all_ok = true;
  qw38::Status status = qw38::Status::ok();
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-120\","
              "\"workload\":\"tails\",\"cases\":[",
              kPrefix);
  for (std::size_t case_index = 0; case_index < 3; ++case_index) {
    const std::size_t prefix = kTailPrefixes[case_index];
    const std::size_t capacity = session_capacity(prefix, 1, 0);
    std::vector<std::size_t> tokens(prefix);
    fill_tokens(&tokens);
    qw38::cuda::SchedulerSession parent;
    qw38::cuda::SchedulerWorkspace parent_ws;
    qw38::cuda::SchedulerSession cand;
    qw38::cuda::SchedulerWorkspace cand_ws;
    qw38::cuda::SchedulerGraphs parent_graphs;
    qw38::cuda::SchedulerGraphs cand_graphs;
    status = parent.create(capacity);
    if (status.is_ok()) status = parent_ws.create(capacity);
    apply_candidate(true, options.packed_kv);
    if (status.is_ok()) status = cand.create(capacity);
    if (status.is_ok()) status = cand_ws.create(capacity);
    clear_candidate();
    if (status.is_ok()) {
      status = create_graphs(model, &parent_ws, &parent_graphs);
    }
    apply_candidate(true, options.packed_kv);
    if (status.is_ok()) status = create_graphs(model, &cand_ws, &cand_graphs);
    std::vector<float> parent_logits(qw38::internal::kVocabularySize);
    std::vector<float> cand_logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> parent_hidden{};
    std::array<float, qw38::internal::kResidualWidth> cand_hidden{};
    clear_candidate();
    if (status.is_ok()) {
      status = prefill(model, tokens, prefix, &parent, &parent_ws,
                       &parent_graphs, parent_logits.data(),
                       parent_hidden.data());
    }
    apply_candidate(true, options.packed_kv);
    if (status.is_ok()) {
      status = prefill(model, tokens, prefix, &cand, &cand_ws, &cand_graphs,
                       cand_logits.data(), cand_hidden.data());
    }
    clear_candidate();
    const bool exact = status.is_ok();
    all_ok = all_ok && exact;
    if (case_index != 0) std::printf(",");
    std::printf("{\"prefix\":%zu,\"ok\":%s,\"exact\":%s}", prefix,
                json_bool(exact), json_bool(exact));
  }
  std::printf("],\"ok\":%s,\"keep\":false}\n", json_bool(all_ok));
  print_counts(0, 1, 2, false);
  return all_ok ? 0 : 1;
}

int run_state_memory(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  (void)model;
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 1, options.capacity);
  qw38::Status status = qw38::Status::ok();
  std::size_t dense_bytes = 0;
  std::size_t packed_bytes = 0;
  {
    apply_candidate(false, options.packed_kv);
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    dense_bytes = session.allocated_bytes();
  }
  if (status.is_ok()) {
    apply_candidate(true, options.packed_kv);
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    packed_bytes = session.allocated_bytes();
  }
  clear_candidate();
  const bool ok = status.is_ok() && packed_bytes > 0 && packed_bytes < dense_bytes;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-120\","
      "\"workload\":\"state-memory\",\"ok\":%s,\"session_bytes\":%zu,"
      "\"dense_session_bytes\":%zu,\"packed_session_bytes\":%zu,"
      "\"bytes_saved\":%zu,\"keep\":false}\n",
      kPrefix, json_bool(ok), packed_bytes, dense_bytes, packed_bytes,
      dense_bytes > packed_bytes ? dense_bytes - packed_bytes : 0);
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

qw38::Status run_session_arm(const qw38::cuda::ResidentModel& model,
                             const std::vector<std::size_t>& tokens,
                             std::size_t prefix, std::size_t outputs,
                             bool candidate, bool prefill_only,
                             const char* packed_kv, std::size_t requested,
                             ArmResult* out) {
  apply_candidate(candidate, packed_kv);
  const std::size_t capacity = session_capacity(prefix, outputs, requested);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
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
  clear_candidate();
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"wall_ms\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,"
      "\"p95_ms\":%.9g,\"ttft_ms\":%.9g}%s",
      name, static_cast<double>(arm.wall_ms), static_cast<double>(arm.tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms),
      static_cast<double>(arm.ttft_ms), last ? "" : ",");
}

int run_session_ab(const qw38::cuda::ResidentModel& model,
                   const Options& options) {
  const bool prefill_only = options.tokens == 0;
  const std::size_t prompt = options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_session_arm(model, tokens, prompt, outputs, false, prefill_only,
                             options.packed_kv, options.capacity, &discarded);
    if (status.is_ok()) {
      status = run_session_arm(model, tokens, prompt, outputs, true,
                               prefill_only, options.packed_kv, options.capacity,
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
    status = run_session_arm(model, tokens, prompt, outputs, ba, prefill_only,
                             options.packed_kv, options.capacity, &first);
    if (status.is_ok()) {
      status = run_session_arm(model, tokens, prompt, outputs, !ba, prefill_only,
                               options.packed_kv, options.capacity, &second);
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
      "%s{\"schema_version\":1,\"task\":\"OPT-120\",\"workload\":\"session-ab\","
      "\"prefix\":%zu,\"prefill\":%s,"
      "\"warmups\":%zu,\"samples\":%zu,\"tokens\":%zu,"
      "\"parent\":\"post113_selected\",\"candidate\":\"%s\","
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"acceptance\",\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true,\"pairs\":[",
      kPrefix, prompt, json_bool(prefill_only), options.warmups, options.samples,
      outputs, options.packed_kv, options.warmups, options.samples);
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
  if (std::strcmp(options.workload, "reference") == 0) {
    return run_reference(options);
  }
  if (std::strcmp(options.workload, "inventory") == 0) {
    return run_inventory(model, options);
  }
  if (std::strcmp(options.workload, "greedy-parity") == 0) {
    return run_greedy_parity(model, options);
  }
  if (std::strcmp(options.workload, "api") == 0) {
    return run_api(model, options);
  }
  if (std::strcmp(options.workload, "format") == 0) {
    return run_format(model, options);
  }
  if (std::strcmp(options.workload, "tails") == 0) {
    return run_tails(model, options);
  }
  if (std::strcmp(options.workload, "cancellation") == 0) {
    return run_api(model, options);
  }
  if (std::strcmp(options.workload, "state-memory") == 0) {
    return run_state_memory(model, options);
  }
  if (std::strcmp(options.workload, "session-ab") == 0) {
    return run_session_ab(model, options);
  }
  return usage(argv[0]);
}
