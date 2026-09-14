#include "attention_decode_path.cuh"
#include "decode_launch_state.cuh"
#include "engine_attribution.h"
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
#include <ctime>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT136_GRAPH_ACCOUNTING_RESULT=";
constexpr char kCounts[] = "QW38_OPT136_NATIVE_COUNTS=";
constexpr std::size_t kDefaultCapacity = 131072;
constexpr std::size_t kRecordCap = 16384;
constexpr std::size_t kDefaultDecodeTokens = 256;
constexpr std::size_t kWindowTokens = 12;
constexpr std::size_t kWindowStarts[] = {0, 122, 244};

struct Options final {
  const char* workload = "identity";
  const char* attribution = "off";
  std::size_t prefix = 128;
  std::size_t tokens = kDefaultDecodeTokens;
  std::size_t capacity = kDefaultCapacity;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

constexpr char kGateEnv[] = "QW38_OPT136_GATE";

std::string gate_path(const char* name) {
  const char* dir = std::getenv(kGateEnv);
  if (dir == nullptr || dir[0] == '\0') return {};
  std::string path(dir);
  if (path.back() != '/') path.push_back('/');
  path += name;
  return path;
}

void write_gate_text(const char* name, const char* text) {
  const std::string path = gate_path(name);
  if (path.empty() || text == nullptr) return;
  FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) return;
  std::fputs(text, file);
  std::fclose(file);
}

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s [--workload identity|unprofiled|graph-capture|node-capture] "
      "[--attribution on|off] [--prefix N] [--tokens N] [--capacity N] "
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

bool legal_prefix(std::size_t prefix) {
  return prefix == 128 || prefix == 2048 || prefix == 8192 || prefix == 32768;
}

int parse_args(int argc, char** argv, Options* options) {
  int model_index = -1;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--attribution") == 0 && index + 1 < argc) {
      options->attribution = argv[++index];
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->prefix)) return usage(argv[0]);
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
  if (!legal_prefix(options->prefix == 0 ? 128 : options->prefix)) {
    std::fprintf(stderr, "invalid --prefix %zu\n", options->prefix);
    return 2;
  }
  return model_index;
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
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
                       qw38::cuda::DecodeAttribution* attribution) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, nullptr, nullptr,
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
                           qw38::cuda::SchedulerGraphs* graphs) {
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  return graphs->create(model, workspace, session);
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

void print_counts(std::size_t warmups, std::size_t samples,
                  std::size_t candidates, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-136\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

void nvtx_step(const char* engine, std::size_t prefix, std::size_t step,
               int layer, const char* op) {
  char label[192];
  std::snprintf(label, sizeof(label),
                "opt136 engine=%s prefix=%zu step=%zu layer=%d op=%s", engine,
                prefix, step, layer, op);
  nvtxRangePushA(label);
}

bool in_window(std::size_t step) {
  for (std::size_t start : kWindowStarts) {
    if (step >= start && step < start + kWindowTokens) return true;
  }
  return false;
}

bool window_begin(std::size_t step) {
  for (std::size_t start : kWindowStarts) {
    if (step == start) return true;
  }
  return false;
}

bool window_end(std::size_t step) {
  for (std::size_t start : kWindowStarts) {
    if (step + 1 == start + kWindowTokens) return true;
  }
  return false;
}

int run_identity(const qw38::cuda::ResidentModel& model,
                 const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity =
      options.capacity == 0 ? kDefaultCapacity : options.capacity;
  std::vector<std::size_t> tokens(prefix + 1);
  fill_tokens(&tokens);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) {
    status = create_graphs(model, &session, &workspace, &graphs);
  }
  workspace.reset_transfer_counters();
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  workspace.reset_transfer_counters();
  float elapsed = 0.0F;
  if (status.is_ok()) {
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    status = run_token(model, tokens[prefix], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs, nullptr);
  }
  const std::uint64_t decode_d2h = workspace.transfer_d2h_bytes_;
  const bool shipping =
      std::strcmp(qw38::cuda::kSelectedExecutionGraphPath,
                  qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0;
  const bool attention =
      qw38::cuda::kSelectedDecodeAttentionCrossoverThreshold == 1024 &&
      qw38::cuda::kSelectedVec128NParts == 16 &&
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax == 4096;
  const bool kv_ok = qw38::cuda::kSelectedPackedKvFormat ==
                     qw38::cuda::PackedKvFormat::kDenseBf16;
  const bool requant_ok =
      std::strcmp(qw38::cuda::kSelectedWeightRequantConfig, "none") == 0;
  const bool gdn_ok = qw38::cuda::kSelectedGdnStateFormat ==
                      qw38::cuda::GdnStateFormat::kFp32;
  const bool lazy_ok = decode_d2h == 4;
  const bool fusion = qw38::cuda::kSelectedMixerNormQ8Fusion &&
                      qw38::cuda::kSelectedFfnNormQ8Fusion &&
                      qw38::cuda::kSelectedLazyOutputMaterialization;
  std::string message;
  json_escape(status.message(), &message);
  const bool ok =
      status.is_ok() && shipping && attention && kv_ok && requant_ok &&
      gdn_ok && lazy_ok && fusion &&
      qw38::cuda::opt110::engine_hooks_ready() &&
      graphs.decode_segment_graph_count() ==
          qw38::cuda::kDecodeSegmentCount *
              static_cast<std::size_t>(qw38::cuda::kDecodeGraphTopologyCount);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-136\",\"workload\":\"identity\","
      "\"ok\":%s,\"parent\":\"post124_plus_opt127_decode_segments8\","
      "\"shipping_execution_graphs\":\"%s\","
      "\"q4_decode\":\"%s\",\"packed_kv\":\"%s\",\"weight_requant\":\"%s\","
      "\"gdn_state\":\"%s\",\"decode_attention\":\"hybrid_crossover\","
      "\"crossover\":%d,\"n_parts\":%d,\"verified_max\":%d,"
      "\"capacity\":%zu,\"decode_segment_graphs\":%zu,"
      "\"lazy\":%s,\"fusion\":%s,\"decode_d2h_bytes\":%llu,"
      "\"opt110_hooks_ready\":%s,\"opt136_diagnostic_markers\":%s,"
      "\"message\":\"%s\",\"claims_throughput\":false,\"keep\":false}\n",
      kPrefix, json_bool(ok), qw38::cuda::kSelectedExecutionGraphPath,
      qw38::cuda::kSelectedQ4DecodePath,
      qw38::cuda::packed_kv_format_ident(qw38::cuda::kSelectedPackedKvFormat),
      qw38::cuda::kSelectedWeightRequantConfig,
      qw38::cuda::gdn_state_format_ident(qw38::cuda::kSelectedGdnStateFormat),
      qw38::cuda::kSelectedDecodeAttentionCrossoverThreshold,
      qw38::cuda::kSelectedVec128NParts,
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax, capacity,
      graphs.decode_segment_graph_count(), json_bool(lazy_ok),
      json_bool(fusion), static_cast<unsigned long long>(decode_d2h),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()),
      json_bool(qw38::cuda::kOpt136DiagnosticIdentityMarkers), message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_trajectory(const qw38::cuda::ResidentModel& model,
                   const Options& options, bool nsys_capture) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs =
      options.tokens == 0 ? kDefaultDecodeTokens : options.tokens;
  const std::size_t capacity =
      options.capacity == 0 ? kDefaultCapacity : options.capacity;
  const bool attribution_on = std::strcmp(options.attribution, "on") == 0;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) {
    status = create_graphs(model, &session, &workspace, &graphs);
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  const auto setup_started = std::chrono::steady_clock::now();
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  const float prefill_ms = wall_ms(setup_started);
  if (!status.is_ok()) return fail_status(status);

  std::vector<qw38::cuda::EngineOpRecord> storage(kRecordCap);
  qw38::cuda::EngineAttribution families{};
  families.records = storage.data();
  families.capacity = storage.size();
  families.record = attribution_on;
  qw38::cuda::copy_cstr(families.engine, sizeof(families.engine), "quartz");
  qw38::cuda::copy_cstr(families.phase, sizeof(families.phase), "decode");
  qw38::cuda::copy_cstr(families.graph_mode, sizeof(families.graph_mode),
                        "decode_segments8");
  cudaDeviceSynchronize();
  if (attribution_on &&
      qw38::cuda::record_sequence_epoch(&families, nullptr) != cudaSuccess) {
    return 1;
  }
  qw38::cuda::DecodeAttribution attribution;
  attribution.record_leaves = attribution_on;
  attribution.families = attribution_on ? &families : nullptr;

  workspace.reset_transfer_counters();
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  const auto decode_started = std::chrono::steady_clock::now();
  float window_ms = 0.0F;
  std::size_t window_steps = 0;
  std::vector<std::int32_t> consumed(outputs, 0);
  bool profiler_started = false;
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    const bool capture_step = nsys_capture && in_window(step);
    if (capture_step && window_begin(step)) {
      cudaDeviceSynchronize();
      nvtxRangePushA("opt136.cpu_bound");
      nvtxRangePop();
      nvtxRangePushA("opt136.gpu_bound");
      nvtxRangePop();
      nvtxRangePushA("opt136.window");
      cudaProfilerStart();
      profiler_started = true;
    }
    nvtx_step("quartz", prefix, step, -1, "eval");
    const std::size_t planned = tokens[prefix + step];
    consumed[step] = static_cast<std::int32_t>(planned);
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    if (status.is_ok()) {
      qw38::cuda::DecodeAttribution* used =
          attribution_on && capture_step ? &attribution : nullptr;
      status = run_token(model, planned, &session, &workspace, logits.data(),
                         hidden.data(), &elapsed, &graphs, used);
    }
    if (status.is_ok()) {
      std::size_t sampled = 0;
      status = qw38::cuda::greedy_sample(session, &sampled);
    }
    nvtxRangePop();
    if (capture_step) {
      window_ms += wall_ms(token_started);
      ++window_steps;
    }
    if (capture_step && window_end(step) && profiler_started) {
      cudaDeviceSynchronize();
      cudaProfilerStop();
      nvtxRangePop();
      profiler_started = false;
    }
  }
  const float decode_only_ms = wall_ms(decode_started);
  const float complete_request_ms = prefill_ms + decode_only_ms;
  if (attribution_on) qw38::cuda::destroy_sequence_epoch(&families);
  if (!status.is_ok()) return fail_status(status);

  std::string message;
  json_escape(status.message(), &message);
  char result[4096];
  std::snprintf(
      result, sizeof(result),
      "%s{\"schema_version\":1,\"task\":\"OPT-136\",\"workload\":\"%s\","
      "\"ok\":%s,\"engine\":\"quartz\",\"prefix\":%zu,\"eval_count\":%zu,"
      "\"emitted_token_count\":0,\"capacity\":%zu,\"window_tokens\":%zu,"
      "\"window_steps\":%zu,\"nsys_capture\":%s,\"attribution\":%s,"
      "\"prefill_ms\":%.9g,\"decode_only_ms\":%.9g,"
      "\"complete_request_ms\":%.9g,\"record_count\":%zu,"
      "\"decode_segment_graphs\":%zu,\"consumed_token_ids\":%zu,"
      "\"message\":\"%s\",\"keep\":false}\n",
      kPrefix, options.workload, json_bool(!families.pool_overflow), prefix,
      outputs, capacity, kWindowTokens, window_steps, json_bool(nsys_capture),
      json_bool(attribution_on), static_cast<double>(prefill_ms),
      static_cast<double>(decode_only_ms),
      static_cast<double>(complete_request_ms), families.count,
      graphs.decode_segment_graph_count(), consumed.size(), message.c_str());
  std::fputs(result, stdout);
  write_gate_text("result.txt", result);
  print_counts(0, 1, 1, false);
  (void)consumed;
  return families.pool_overflow ? 1 : 0;
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
  std::printf("independently_restored=true same_binary=true\n");
  if (std::strcmp(options.workload, "identity") == 0) {
    return run_identity(model, options);
  }
  if (std::strcmp(options.workload, "unprofiled") == 0) {
    return run_trajectory(model, options, false);
  }
  if (std::strcmp(options.workload, "graph-capture") == 0 ||
      std::strcmp(options.workload, "node-capture") == 0) {
    return run_trajectory(model, options, true);
  }
  return usage(argv[0]);
}
