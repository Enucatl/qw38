#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "opt110_engine_hook.cuh"
#include "opt121_weight_requant.cuh"
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

#include "model.h"
#include "quant.h"
#include "scheduler.h"
#include "tensor.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT121_WEIGHT_TRAFFIC_RESULT=";
constexpr char kCounts[] = "QW38_OPT121_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 8192;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;

struct Options final {
  const char* workload = "inventory";
  const char* weight_requant = "q8_to_q4k";
  const char* execution_graphs = "ffn_only";
  std::size_t prefix = 128;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
  std::size_t capacity = 0;
};

struct ArmResult final {
  float wall_ms = 0.0F;
  float tok_s = 0.0F;
  float p50_ms = 0.0F;
  float p95_ms = 0.0F;
  float ttft_ms = 0.0F;
};

struct HostModel final {
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelWeights weights;
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
               "[--weight-requant none|q8_to_q4k|q8_q6_to_q4k|fp4_study] "
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
    } else if (std::strcmp(arg, "--weight-requant") == 0 && index + 1 < argc) {
      options->weight_requant = argv[++index];
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
  if (!qw38::cuda::legal_weight_requant_ident(options->weight_requant)) {
    std::fprintf(stderr, "invalid --weight-requant %s\n", options->weight_requant);
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

qw38::Status bind_host(const char* path, HostModel* host) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  if (status.is_ok()) status = host->mapping.open(path);
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, host->mapping,
                                                &host->weights);
  }
  return status;
}

void apply_candidate(bool candidate, const char* ident) {
  if (candidate) {
    qw38::cuda::set_weight_requant_override(
        qw38::cuda::weight_requant_from_ident(ident));
  } else {
    qw38::cuda::set_weight_requant_override(qw38::cuda::WeightRequantConfig::kNone);
  }
}

void clear_candidate() { qw38::cuda::clear_weight_requant_override(); }

qw38::Status upload_model(HostModel* host, qw38::cuda::ResidentModel* model,
                          bool candidate, const char* ident) {
  *model = qw38::cuda::ResidentModel();
  apply_candidate(candidate, ident);
  const qw38::Status status =
      model->upload(host->weights, host->mapping.data(), host->mapping.size());
  clear_candidate();
  return status;
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
      "%s{\"schema_version\":1,\"task\":\"OPT-121\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

int run_reference(const Options& options) {
  float values[256];
  for (int i = 0; i < 256; ++i) {
    values[i] = static_cast<float>(i - 128) * 0.015625F;
  }
  std::uint8_t block[144];
  float back[256];
  qw38::Status status =
      qw38::internal::encode_q4_k(values, 256, block, 144);
  if (status.is_ok()) {
    status = qw38::internal::decode_q4_k(block, 144, back, 256);
  }
  double mse = 0.0;
  bool finite = true;
  if (status.is_ok()) {
    for (int i = 0; i < 256; ++i) {
      finite = finite && std::isfinite(back[i]);
      const double err = static_cast<double>(back[i] - values[i]);
      mse += err * err;
    }
    mse /= 256.0;
  }
  int sm = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    sm = prop.major * 10 + prop.minor;
  }
  const bool fp4_native_mma = sm >= 120;
  const bool fp4_stopped =
      std::strcmp(options.weight_requant, "fp4_study") == 0;
  const bool capability_absent = true;
  const bool ok = status.is_ok() && finite && mse < 0.05 &&
                  qw38::cuda::legal_weight_requant_ident(options.weight_requant);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-121\",\"workload\":\"reference\","
      "\"ok\":%s,\"q4k_roundtrip_mse\":%.9g,\"finite\":%s,"
      "\"sm\":%d,\"fp4_native_mma_sm120\":%s,\"fp4_study\":%s,"
      "\"fp4_capability_absent\":%s,\"fp4_not_relabeled_q4k\":true,"
      "\"ds4_mxfp4_source_required\":true,\"encoding\":\"%s\","
      "\"policy\":\"%s\",\"calibration\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"max_configs\":3,\"keep\":false}\n",
      kPrefix, json_bool(ok), mse, json_bool(finite), sm,
      json_bool(fp4_native_mma), json_bool(fp4_stopped),
      json_bool(capability_absent), qw38::cuda::kOpt121Encoding,
      qw38::cuda::kOpt121PolicyId, qw38::cuda::kOpt121CalibrationId,
      qw38::cuda::kOpt121GgufSha);
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_inventory(HostModel* host, const Options& options) {
  const char* configs[3] = {"q8_to_q4k", "q8_q6_to_q4k", "fp4_study"};
  std::size_t converted[3] = {0, 0, 0};
  std::size_t src_bytes[3] = {0, 0, 0};
  std::size_t dst_bytes[3] = {0, 0, 0};
  float conversion_ms[3] = {0.0F, 0.0F, 0.0F};
  std::size_t cand_resident = 0;
  std::size_t parent_resident = 0;
  std::size_t selected_index = 0;
  bool configs_ok = true;
  {
    qw38::cuda::ResidentModel parent;
    const qw38::Status status =
        upload_model(host, &parent, false, options.weight_requant);
    if (!status.is_ok()) return fail_status(status);
    parent_resident = parent.resident_bytes();
  }
  for (int index = 0; index < 3; ++index) {
    if (std::strcmp(options.weight_requant, configs[index]) == 0) {
      selected_index = static_cast<std::size_t>(index);
    }
    qw38::cuda::ResidentModel candidate;
    const qw38::Status status =
        upload_model(host, &candidate, true, configs[index]);
    if (!status.is_ok()) return fail_status(status);
    converted[index] = candidate.converted_tensor_count();
    src_bytes[index] = candidate.converted_src_bytes();
    dst_bytes[index] = candidate.converted_dst_bytes();
    conversion_ms[index] = candidate.conversion_milliseconds();
    if (index == 0) cand_resident = candidate.resident_bytes();
    const bool expected_q4 =
        std::strcmp(configs[index], "fp4_study") == 0
            ? converted[index] == 0 &&
                  candidate.layer(0).gdn.packed_qkv.kind ==
                      qw38::cuda::QuantKind::kQ8_0
            : converted[index] > 0 &&
                  candidate.layer(0).gdn.packed_qkv.kind ==
                      qw38::cuda::QuantKind::kQ4K &&
                  candidate.layer(0).gdn.alpha.kind ==
                      qw38::cuda::QuantKind::kQ8_0 &&
                  dst_bytes[index] <= src_bytes[index];
    configs_ok = configs_ok && expected_q4;
  }
  const bool ok = configs_ok;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-121\",\"workload\":\"inventory\","
      "\"ok\":%s,\"parent\":\"post113_selected\",\"execution_graphs\":\"ffn_only\","
      "\"primary_target\":\"decode\",\"candidate\":\"%s\","
      "\"converted_tensors\":%zu,\"src_bytes\":%zu,\"dst_bytes\":%zu,"
      "\"bytes_saved\":%zu,\"conversion_ms\":%.9g,"
      "\"parent_resident_bytes\":%zu,\"candidate_resident_bytes\":%zu,"
      "\"q8_to_q4k_tensors\":%zu,\"q8_q6_to_q4k_tensors\":%zu,"
      "\"fp4_study_tensors\":%zu,\"fp4_capability_absent\":true,"
      "\"max_configs\":3,\"no_dense_shadow\":true,\"irreversible_loss\":true,"
      "\"embedding_excluded_from_traffic\":true,\"keep\":false}\n",
      kPrefix, json_bool(ok), options.weight_requant,
      converted[selected_index], src_bytes[selected_index],
      dst_bytes[selected_index],
      src_bytes[selected_index] > dst_bytes[selected_index]
          ? src_bytes[selected_index] - dst_bytes[selected_index]
          : 0,
      static_cast<double>(conversion_ms[selected_index]), parent_resident,
      cand_resident, converted[0], converted[1], converted[2]);
  print_counts(0, 1, 3, false);
  return ok ? 0 : 1;
}

int run_greedy_parity(HostModel* host, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  std::vector<float> parent_logits(qw38::internal::kVocabularySize);
  std::vector<float> cand_logits(qw38::internal::kVocabularySize);
  std::size_t parent_token = 0;
  std::size_t cand_token = 0;
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::ResidentModel parent_model;
    status = upload_model(host, &parent_model, false, options.weight_requant);
    qw38::cuda::SchedulerSession parent;
    qw38::cuda::SchedulerWorkspace parent_ws;
    qw38::cuda::SchedulerGraphs parent_graphs;
    if (status.is_ok()) status = parent.create(capacity);
    if (status.is_ok()) status = parent_ws.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(parent_model, &parent_ws, &parent_graphs);
    }
    std::array<float, qw38::internal::kResidualWidth> parent_hidden{};
    if (status.is_ok()) {
      status = prefill(parent_model, tokens, prefix, &parent, &parent_ws,
                       &parent_graphs, parent_logits.data(),
                       parent_hidden.data());
    }
    if (status.is_ok()) status = qw38::cuda::greedy_sample(parent, &parent_token);
    if (!status.is_ok()) return fail_status(status);
  }
  {
    qw38::cuda::ResidentModel cand_model;
    status = upload_model(host, &cand_model, true, options.weight_requant);
    qw38::cuda::SchedulerSession cand;
    qw38::cuda::SchedulerWorkspace cand_ws;
    qw38::cuda::SchedulerGraphs cand_graphs;
    if (status.is_ok()) status = cand.create(capacity);
    if (status.is_ok()) status = cand_ws.create(capacity);
    if (status.is_ok()) status = create_graphs(cand_model, &cand_ws, &cand_graphs);
    std::array<float, qw38::internal::kResidualWidth> cand_hidden{};
    if (status.is_ok()) {
      status = prefill(cand_model, tokens, prefix, &cand, &cand_ws, &cand_graphs,
                       cand_logits.data(), cand_hidden.data());
    }
    if (status.is_ok()) status = qw38::cuda::greedy_sample(cand, &cand_token);
    if (!status.is_ok()) return fail_status(status);
  }
  bool finite = true;
  for (float value : cand_logits) finite = finite && std::isfinite(value);
  const bool exact =
      std::memcmp(parent_logits.data(), cand_logits.data(),
                  parent_logits.size() * sizeof(float)) == 0;
  const bool ok = finite;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-121\","
      "\"workload\":\"greedy-parity\",\"ok\":%s,\"exact\":%s,"
      "\"finite\":%s,\"greedy_equal\":%s,\"parent_token\":%zu,"
      "\"candidate_token\":%zu,\"approximate\":true,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(exact), json_bool(finite),
      json_bool(parent_token == cand_token), parent_token, cand_token);
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_api(HostModel* host, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 3, options.capacity);
  std::vector<std::size_t> tokens(prefix + 3);
  fill_tokens(&tokens);
  qw38::cuda::ResidentModel model;
  qw38::Status status = upload_model(host, &model, true, options.weight_requant);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  std::vector<float> before(qw38::internal::kVocabularySize);
  if (status.is_ok()) {
    status = session.copy_last_outputs(before.data(), before.size(), hidden.data(),
                                       hidden.size());
  }
  std::vector<float> again(qw38::internal::kVocabularySize);
  if (status.is_ok()) {
    status = session.copy_last_outputs(again.data(), again.size(), hidden.data(),
                                       hidden.size());
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
  const char* tmp = "/tmp/qw38-opt121-ckpt.bin";
  if (status.is_ok()) status = session.save_checkpoint(tmp);
  qw38::cuda::SchedulerSession restored;
  if (status.is_ok()) status = restored.create(capacity);
  if (status.is_ok()) status = restored.restore_checkpoint(tmp, &workspace);
  bool equal = false;
  if (status.is_ok()) status = restored.state_equals(session, &equal);
  std::remove(tmp);
  struct PollContext final {
    std::size_t calls = 0;
    std::size_t stop_after = 1;
  } poll;
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
  const bool ok = status.is_ok() && repeat_equal && eval_changed && equal &&
                  cancel_isolated;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-121\",\"workload\":\"api\","
      "\"ok\":%s,\"repeat_logits_equal\":%s,\"eval_changes_logits\":%s,"
      "\"restore_equals\":%s,\"cancel_isolated\":%s,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(repeat_equal), json_bool(eval_changed),
      json_bool(equal), json_bool(cancel_isolated));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_format(HostModel* host, const Options& options) {
  qw38::cuda::ResidentModel model;
  qw38::Status status = upload_model(host, &model, true, options.weight_requant);
  if (!status.is_ok()) return fail_status(status);
  const bool converted =
      std::strcmp(options.weight_requant, "q8_to_q4k") != 0 ||
      (model.converted_tensor_count() > 0 &&
       model.layer(0).gdn.packed_qkv.kind == qw38::cuda::QuantKind::kQ4K &&
       model.layer(0).gdn.alpha.kind == qw38::cuda::QuantKind::kQ8_0);
  const bool identity =
      std::strcmp(model.weight_requant_config(), options.weight_requant) == 0 ||
      std::strcmp(options.weight_requant, "fp4_study") == 0;
  const bool ok = converted && identity;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-121\",\"workload\":\"format\","
      "\"ok\":%s,\"config\":\"%s\",\"applied\":\"%s\","
      "\"converted_tensors\":%zu,\"conversion_ms\":%.9g,"
      "\"cache_identity\":\"%s/%s/%s\",\"gguf_unmodified\":true,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), options.weight_requant, model.weight_requant_config(),
      model.converted_tensor_count(),
      static_cast<double>(model.conversion_milliseconds()),
      qw38::cuda::kOpt121GgufSha, qw38::cuda::kOpt121PolicyId,
      qw38::cuda::kOpt121Encoding);
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_state_memory(HostModel* host, const Options& options) {
  std::size_t parent_resident = 0;
  std::size_t cand_resident = 0;
  std::size_t src_bytes = 0;
  std::size_t dst_bytes = 0;
  std::size_t session_bytes = 0;
  {
    qw38::cuda::ResidentModel parent;
    const qw38::Status status =
        upload_model(host, &parent, false, options.weight_requant);
    if (!status.is_ok()) return fail_status(status);
    parent_resident = parent.resident_bytes();
  }
  {
    qw38::cuda::ResidentModel candidate;
    qw38::Status status =
        upload_model(host, &candidate, true, options.weight_requant);
    const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
    const std::size_t capacity = session_capacity(prefix, 1, options.capacity);
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    if (status.is_ok()) status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (!status.is_ok()) return fail_status(status);
    cand_resident = candidate.resident_bytes();
    src_bytes = candidate.converted_src_bytes();
    dst_bytes = candidate.converted_dst_bytes();
    session_bytes = session.allocated_bytes();
  }
  const bool ok = cand_resident > 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-121\","
      "\"workload\":\"state-memory\",\"ok\":%s,\"session_bytes\":%zu,"
      "\"parent_resident_bytes\":%zu,\"candidate_resident_bytes\":%zu,"
      "\"converted_src_bytes\":%zu,\"converted_dst_bytes\":%zu,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), session_bytes, parent_resident, cand_resident,
      src_bytes, dst_bytes);
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

qw38::Status run_session_arm(HostModel* host, const std::vector<std::size_t>& tokens,
                             std::size_t prefix, std::size_t outputs,
                             bool candidate, bool prefill_only,
                             const char* ident, std::size_t requested,
                             ArmResult* out) {
  qw38::cuda::ResidentModel model;
  qw38::Status status = upload_model(host, &model, candidate, ident);
  const std::size_t capacity = session_capacity(prefix, outputs, requested);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = session.create(capacity);
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

int run_session_ab(HostModel* host, const Options& options) {
  const bool prefill_only = options.tokens == 0;
  const std::size_t prompt = options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_session_arm(host, tokens, prompt, outputs, false, prefill_only,
                             options.weight_requant, options.capacity,
                             &discarded);
    if (status.is_ok()) {
      status = run_session_arm(host, tokens, prompt, outputs, true, prefill_only,
                               options.weight_requant, options.capacity,
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
    status = run_session_arm(host, tokens, prompt, outputs, ba, prefill_only,
                             options.weight_requant, options.capacity, &first);
    if (status.is_ok()) {
      status = run_session_arm(host, tokens, prompt, outputs, !ba, prefill_only,
                               options.weight_requant, options.capacity, &second);
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
      "%s{\"schema_version\":1,\"task\":\"OPT-121\",\"workload\":\"session-ab\","
      "\"prefix\":%zu,\"prefill\":%s,"
      "\"warmups\":%zu,\"samples\":%zu,\"tokens\":%zu,"
      "\"parent\":\"post113_selected\",\"candidate\":\"%s\","
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"acceptance\",\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true,\"pairs\":[",
      kPrefix, prompt, json_bool(prefill_only), options.warmups, options.samples,
      outputs, options.weight_requant, options.warmups, options.samples);
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
  if (std::strcmp(options.workload, "reference") == 0) {
    return run_reference(options);
  }
  HostModel host;
  const qw38::Status status = bind_host(argv[model_index], &host);
  if (!status.is_ok()) return fail_status(status);
  if (std::strcmp(options.workload, "inventory") == 0) {
    return run_inventory(&host, options);
  }
  if (std::strcmp(options.workload, "greedy-parity") == 0) {
    return run_greedy_parity(&host, options);
  }
  if (std::strcmp(options.workload, "api") == 0) {
    return run_api(&host, options);
  }
  if (std::strcmp(options.workload, "format") == 0) {
    return run_format(&host, options);
  }
  if (std::strcmp(options.workload, "state-memory") == 0) {
    return run_state_memory(&host, options);
  }
  if (std::strcmp(options.workload, "session-ab") == 0) {
    return run_session_ab(&host, options);
  }
  return usage(argv[0]);
}
