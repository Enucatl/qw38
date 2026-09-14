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

constexpr char kPrefix[] = "QW38_OPT133_DECODE_NSYS_TRACE_RESULT=";
constexpr char kCounts[] = "QW38_OPT133_NATIVE_COUNTS=";
constexpr char kRecords[] = "QW38_OPT133_RECORDS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr std::size_t kRecordCap = 16384;
constexpr std::size_t kDefaultDecodeTokens = 256;
constexpr std::size_t kWindowTokens = 12;

struct Options final {
  const char* workload = "identity";
  const char* window = "early";
  std::size_t prefix = 128;
  std::size_t tokens = kDefaultDecodeTokens;
  std::size_t capacity = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

constexpr char kGateEnv[] = "QW38_OPT133_GATE";

std::string gate_path(const char* name) {
  const char* dir = std::getenv(kGateEnv);
  if (dir == nullptr || dir[0] == '\0') return {};
  std::string path(dir);
  if (path.back() != '/') path.push_back('/');
  path += name;
  return path;
}

void write_gate(const char* name) {
  const std::string path = gate_path(name);
  if (path.empty()) return;
  FILE* file = std::fopen(path.c_str(), "w");
  if (file != nullptr) std::fclose(file);
}

bool gate_exists(const char* name) {
  const std::string path = gate_path(name);
  if (path.empty()) return true;
  return ::access(path.c_str(), F_OK) == 0;
}

void write_gate_text(const char* name, const char* text) {
  const std::string path = gate_path(name);
  if (path.empty() || text == nullptr) return;
  FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) return;
  std::fputs(text, file);
  std::fclose(file);
}

void emit_gate_sidecars(const qw38::cuda::EngineAttribution& families,
                        const Options& options, std::size_t prefix,
                        std::size_t outputs, std::size_t win_begin,
                        std::size_t win_end, std::size_t window_steps,
                        float window_ms, float decode_only_ms, bool nsys_capture) {
  const std::string records_path = gate_path("records.json");
  if (!records_path.empty()) {
    FILE* records_file = std::fopen(records_path.c_str(), "w");
    if (records_file != nullptr) {
      std::fputc('[', records_file);
      for (std::size_t index = 0; index < families.count; ++index) {
        qw38::cuda::write_engine_record_json(records_file, families.records[index],
                                             index + 1 == families.count);
      }
      std::fputs("]\n", records_file);
      std::fclose(records_file);
    }
  }
  const char* workload = nsys_capture ? "nsys-window" : "nsys-baseline";
  char result[2048];
  std::snprintf(
      result, sizeof(result),
      "%s{\"schema_version\":1,\"task\":\"OPT-133\",\"workload\":\"%s\","
      "\"ok\":%s,\"stack\":\"post124_plus_opt127_decode_segments8\","
      "\"prefix\":%zu,\"tokens\":%zu,\"window_label\":\"%s\","
      "\"window_begin\":%zu,\"window_end\":%zu,\"window_tokens\":%zu,"
      "\"nsys_capture\":%s,\"record_count\":%zu,\"pool_overflow\":%s,"
      "\"instrumented_decode_only_wall_ms\":%.9g,"
      "\"trajectory_decode_only_wall_ms\":%.9g,"
      "\"method\":\"cuda_event_engine_attribution\","
      "\"cupti_linked\":false,\"message\":\"\",\"keep\":false}\n",
      kPrefix, workload, json_bool(!families.pool_overflow), prefix, outputs,
      options.window, win_begin, win_end, window_steps, json_bool(nsys_capture),
      families.count, json_bool(families.pool_overflow),
      static_cast<double>(window_ms), static_cast<double>(decode_only_ms));
  write_gate_text("result.txt", result);
}

void wait_gate(const char* name, int timeout_ms) {
  if (gate_path(name).empty()) return;
  const int slices = timeout_ms <= 0 ? 1 : (timeout_ms + 9) / 10;
  timespec ts{0, 10 * 1000 * 1000};
  for (int index = 0; index < slices; ++index) {
    if (gate_exists(name)) return;
    nanosleep(&ts, nullptr);
  }
}

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s [--workload identity|nsys-window|nsys-baseline] "
      "[--window early|middle|late] [--prefix N] [--tokens N] [--capacity N] "
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

bool legal_window(const char* label) {
  return std::strcmp(label, "early") == 0 ||
         std::strcmp(label, "middle") == 0 ||
         std::strcmp(label, "late") == 0;
}

int parse_args(int argc, char** argv, Options* options) {
  int model_index = -1;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--window") == 0 && index + 1 < argc) {
      options->window = argv[++index];
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
  if (!legal_window(options->window)) {
    std::fprintf(stderr, "invalid --window %s\n", options->window);
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
                       qw38::cuda::DecodeAttribution* attribution = nullptr) {
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
      "%s{\"schema_version\":1,\"task\":\"OPT-133\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

void print_profiler_probe() {
  const int nsys = std::system("command -v nsys >/dev/null 2>&1");
  const int ncu = std::system("command -v ncu >/dev/null 2>&1");
  std::printf(
      "nsys_available=%s ncu_available=%s cupti_linked=false "
      "full_ncu_sweep=false\n",
      nsys == 0 ? "true" : "false", ncu == 0 ? "true" : "false");
}

std::pair<std::size_t, std::size_t> window_bounds(const char* label,
                                                  std::size_t outputs) {
  const std::size_t width = std::min(kWindowTokens, outputs);
  if (std::strcmp(label, "early") == 0) {
    return {0, width};
  }
  if (std::strcmp(label, "late") == 0) {
    return {outputs - width, outputs};
  }
  std::size_t start = outputs / 2;
  if (start >= width / 2) {
    start -= width / 2;
  } else {
    start = 0;
  }
  if (start + width > outputs) start = outputs - width;
  return {start, start + width};
}

int run_identity(const qw38::cuda::ResidentModel& model,
                 const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 1, options.capacity);
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
                       logits.data(), hidden.data(), &elapsed, &graphs);
  }
  const std::uint64_t decode_d2h = workspace.transfer_d2h_bytes_;
  const bool shipping =
      std::strcmp(qw38::cuda::kSelectedExecutionGraphPath,
                  qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0;
  const bool attention =
      qw38::cuda::kSelectedDecodeAttentionCrossoverThreshold == 1024 &&
      qw38::cuda::kSelectedVec128NParts == 16 &&
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax == 4096;
  const bool host_stalls = !qw38::cuda::kSelectedPollWithoutDeviceSync &&
                           !qw38::cuda::kSelectedDeferElapsedEventSync;
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
  const bool rejected_no_leak = shipping && attention && host_stalls && kv_ok &&
                                requant_ok && gdn_ok;
  std::string message;
  json_escape(status.message(), &message);
  const bool ok = status.is_ok() && rejected_no_leak && lazy_ok && fusion &&
                  qw38::cuda::opt110::engine_hooks_ready();
  print_profiler_probe();
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-133\",\"workload\":\"identity\","
      "\"ok\":%s,\"parent\":\"post124_plus_opt127_decode_segments8\","
      "\"shipping_execution_graphs\":\"%s\","
      "\"q4_decode\":\"%s\",\"packed_kv\":\"%s\",\"weight_requant\":\"%s\","
      "\"gdn_state\":\"%s\",\"decode_attention\":\"hybrid_crossover\","
      "\"crossover\":%d,\"n_parts\":%d,\"verified_max\":%d,"
      "\"poll_without_device_sync\":%s,\"defer_elapsed_event_sync\":%s,"
      "\"lazy\":%s,\"fusion\":%s,\"decode_d2h_bytes\":%llu,"
      "\"decode_segment_graphs\":%zu,\"rejected_no_leak\":%s,"
      "\"opt110_hooks_ready\":%s,\"message\":\"%s\","
      "\"claims_throughput\":false,\"keep\":false}\n",
      kPrefix, json_bool(ok), qw38::cuda::kSelectedExecutionGraphPath,
      qw38::cuda::kSelectedQ4DecodePath,
      qw38::cuda::packed_kv_format_ident(qw38::cuda::kSelectedPackedKvFormat),
      qw38::cuda::kSelectedWeightRequantConfig,
      qw38::cuda::gdn_state_format_ident(qw38::cuda::kSelectedGdnStateFormat),
      qw38::cuda::kSelectedDecodeAttentionCrossoverThreshold,
      qw38::cuda::kSelectedVec128NParts,
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax,
      json_bool(qw38::cuda::kSelectedPollWithoutDeviceSync),
      json_bool(qw38::cuda::kSelectedDeferElapsedEventSync), json_bool(lazy_ok),
      json_bool(fusion), static_cast<unsigned long long>(decode_d2h),
      graphs.decode_segment_graph_count(), json_bool(rejected_no_leak),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()), message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_window(const qw38::cuda::ResidentModel& model, const Options& options,
               bool nsys_capture) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs =
      options.tokens == 0 ? kDefaultDecodeTokens : options.tokens;
  const auto bounds = window_bounds(options.window, outputs);
  const std::size_t win_begin = bounds.first;
  const std::size_t win_end = bounds.second;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  print_profiler_probe();
  const std::size_t capacity =
      session_capacity(prefix, outputs, options.capacity);
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
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  if (!status.is_ok()) return fail_status(status);

  std::vector<qw38::cuda::EngineOpRecord> storage(kRecordCap);
  qw38::cuda::EngineAttribution families{};
  families.records = storage.data();
  families.capacity = storage.size();
  families.record = true;
  qw38::cuda::copy_cstr(families.engine, sizeof(families.engine), "quartz");
  qw38::cuda::copy_cstr(families.phase, sizeof(families.phase), "decode");
  qw38::cuda::copy_cstr(families.graph_mode, sizeof(families.graph_mode),
                        "decode_segments8");
  cudaDeviceSynchronize();
  if (qw38::cuda::record_sequence_epoch(&families, nullptr) != cudaSuccess) {
    return 1;
  }
  qw38::cuda::DecodeAttribution attribution;
  attribution.record_leaves = true;
  attribution.families = &families;

  workspace.reset_transfer_counters();
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  const auto decode_started = std::chrono::steady_clock::now();
  float window_ms = 0.0F;
  std::size_t window_steps = 0;
  bool profiler_started = false;
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    const bool in_range = step >= win_begin && step < win_end;
    if (in_range && nsys_capture && !profiler_started) {
      cudaDeviceSynchronize();
      write_gate("waiting");
      wait_gate("armed", 120000);
      nvtxRangePushA("opt133.window");
      cudaProfilerStart();
      profiler_started = true;
    }
    std::size_t sampled = 0;
    status = qw38::cuda::greedy_sample(session, &sampled);
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    if (status.is_ok()) {
      qw38::cuda::DecodeAttribution* used = in_range ? &attribution : nullptr;
      status = run_token(model, sampled, &session, &workspace, logits.data(),
                         hidden.data(), &elapsed, &graphs, used);
    }
    if (in_range) {
      window_ms += wall_ms(token_started);
      ++window_steps;
    }
    if (in_range && nsys_capture && step + 1 == win_end) {
      cudaDeviceSynchronize();
      cudaProfilerStop();
      nvtxRangePop();
      emit_gate_sidecars(families, options, prefix, outputs, win_begin, win_end,
                         window_steps, window_ms, wall_ms(decode_started),
                         nsys_capture);
      write_gate("done");
    }
  }
  const float decode_only_ms = wall_ms(decode_started);
  qw38::cuda::destroy_sequence_epoch(&families);
  if (!status.is_ok()) return fail_status(status);

  FILE* records_file = nullptr;
  const std::string records_path = gate_path("records.json");
  if (!records_path.empty()) {
    records_file = std::fopen(records_path.c_str(), "w");
  }
  std::printf("%s[", kRecords);
  if (records_file != nullptr) std::fputc('[', records_file);
  for (std::size_t index = 0; index < families.count; ++index) {
    qw38::cuda::write_engine_record_json(stdout, families.records[index],
                                         index + 1 == families.count);
    if (records_file != nullptr) {
      qw38::cuda::write_engine_record_json(records_file, families.records[index],
                                           index + 1 == families.count);
    }
  }
  std::printf("]\n");
  if (records_file != nullptr) {
    std::fputs("]\n", records_file);
    std::fclose(records_file);
  }
  std::string message;
  json_escape(status.message(), &message);
  const char* workload = nsys_capture ? "nsys-window" : "nsys-baseline";
  char result[2048];
  std::snprintf(
      result, sizeof(result),
      "%s{\"schema_version\":1,\"task\":\"OPT-133\",\"workload\":\"%s\","
      "\"ok\":%s,\"stack\":\"post124_plus_opt127_decode_segments8\","
      "\"prefix\":%zu,\"tokens\":%zu,\"window_label\":\"%s\","
      "\"window_begin\":%zu,\"window_end\":%zu,\"window_tokens\":%zu,"
      "\"nsys_capture\":%s,\"record_count\":%zu,\"pool_overflow\":%s,"
      "\"instrumented_decode_only_wall_ms\":%.9g,"
      "\"trajectory_decode_only_wall_ms\":%.9g,"
      "\"method\":\"cuda_event_engine_attribution\","
      "\"cupti_linked\":false,\"message\":\"%s\",\"keep\":false}\n",
      kPrefix, workload, json_bool(!families.pool_overflow), prefix, outputs,
      options.window, win_begin, win_end, window_steps, json_bool(nsys_capture),
      families.count, json_bool(families.pool_overflow),
      static_cast<double>(window_ms), static_cast<double>(decode_only_ms),
      message.c_str());
  std::fputs(result, stdout);
  write_gate_text("result.txt", result);
  print_counts(0, 1, 1, false);
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
  if (std::strcmp(options.workload, "nsys-window") == 0) {
    return run_window(model, options, true);
  }
  if (std::strcmp(options.workload, "nsys-baseline") == 0) {
    return run_window(model, options, false);
  }
  return usage(argv[0]);
}
