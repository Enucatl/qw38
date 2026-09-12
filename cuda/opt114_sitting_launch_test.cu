#include "engine_attribution.h"
#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "test_tier.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT114_SITTING_LAUNCH_RESULT=";
constexpr char kCounts[] = "QW38_OPT114_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr std::size_t kRecordCap = 8192;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;

struct Options final {
  const char* workload = "aa-control";
  const char* execution_graphs = "ffn_only";
  std::size_t prefix = 128;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload aa-control|launch-overhead|graph-ab|graph-bench] "
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

std::size_t session_capacity(std::size_t prefix, std::size_t outputs) {
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
                       qw38::cuda::DecodeAttribution* attribution) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, nullptr, nullptr,
      qw38::cuda::PointwisePath::kFused, graphs, attribution);
}

struct ArmResult final {
  float wall_ms = 0.0F;
  float tok_s = 0.0F;
  float p50_ms = 0.0F;
  float p95_ms = 0.0F;
};

qw38::Status run_prefill(const qw38::cuda::ResidentModel& model,
                         const std::vector<std::size_t>& tokens,
                         std::size_t prompt, ArmResult* out) {
  const std::size_t capacity = session_capacity(prompt, 0);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return status;
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  const auto started = std::chrono::steady_clock::now();
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), prompt, &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &sync, nullptr, &graphs);
  out->wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - started)
                                        .count());
  out->tok_s = out->wall_ms > 0.0F
                   ? static_cast<float>(prompt) * 1000.0F / out->wall_ms
                   : 0.0F;
  out->p50_ms = out->wall_ms;
  out->p95_ms = out->wall_ms;
  return status;
}

qw38::Status run_decode(const qw38::cuda::ResidentModel& model,
                        const std::vector<std::size_t>& tokens,
                        std::size_t prefix, std::size_t outputs, ArmResult* out,
                        qw38::cuda::DecodeAttribution* attribution = nullptr) {
  const std::size_t capacity = session_capacity(prefix, outputs);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return status;
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), prefix, &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &sync, nullptr, &graphs);
  if (!status.is_ok()) return status;
  std::vector<float> latencies;
  latencies.reserve(outputs);
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; step < outputs; ++step) {
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs,
                       attribution);
    if (!status.is_ok()) return status;
    latencies.push_back(static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - token_started)
            .count()));
  }
  out->wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - started)
                                        .count());
  out->tok_s = out->wall_ms > 0.0F
                   ? static_cast<float>(outputs) * 1000.0F / out->wall_ms
                   : 0.0F;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"wall_ms\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,"
      "\"p95_ms\":%.9g}%s",
      name, static_cast<double>(arm.wall_ms), static_cast<double>(arm.tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms),
      last ? "" : ",");
}

int run_aa_control(const qw38::cuda::ResidentModel& model,
                   const Options& options) {
  const bool prefill = options.prefix == 4096;
  const std::size_t prompt = prefill ? 4096 : options.prefix;
  const std::size_t outputs = prefill ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  qw38::cuda::ExecutionGraphPathScope scope(options.execution_graphs);
  qw38::Status status = qw38::Status::ok();

  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    if (prefill) {
      status = run_prefill(model, tokens, prompt, &discarded);
    } else {
      status = run_decode(model, tokens, prompt, outputs, &discarded);
    }
    if (!status.is_ok()) return fail_status(status);
    std::printf("quartz_warmup=%zu tok_s=%.9g wall_ms=%.9g\n", warmup,
                static_cast<double>(discarded.tok_s),
                static_cast<double>(discarded.wall_ms));
  }

  std::vector<ArmResult> arm_a(options.samples);
  std::vector<ArmResult> arm_b(options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    const bool ba = (pair % 2) == 1;
    ArmResult first;
    ArmResult second;
    if (prefill) {
      status = run_prefill(model, tokens, prompt, &first);
      if (status.is_ok()) status = run_prefill(model, tokens, prompt, &second);
    } else {
      status = run_decode(model, tokens, prompt, outputs, &first);
      if (status.is_ok()) {
        status = run_decode(model, tokens, prompt, outputs, &second);
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
      "%s{\"schema_version\":1,\"task\":\"OPT-114\",\"workload\":\"aa-control\","
      "\"prefix\":%zu,\"prefill\":%s,\"warmups\":%zu,\"samples\":%zu,"
      "\"tokens\":%zu,\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\",\"keep\":false,"
      "\"execution_graphs\":\"%s\",\"same_binary\":true,\"pairs\":[",
      kPrefix, options.prefix, json_bool(prefill), options.warmups,
      options.samples, outputs, options.warmups, options.samples,
      qw38::cuda::effective_execution_graph_path());
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    if (pair != 0) std::printf(",");
    std::printf("{\"sample_index\":%zu,\"order\":\"%s\",", pair,
                (pair % 2) == 1 ? "BA" : "AB");
    print_arm("A", arm_a[pair], false);
    print_arm("B", arm_b[pair], true);
    std::printf("}");
  }
  std::printf("]}\n");
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-114\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":2,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":false,"
      "\"sample_ids\":[",
      kCounts, options.warmups, options.samples);
  for (std::size_t index = 0; index < options.samples; ++index) {
    if (index != 0) std::printf(",");
    std::printf("%zu", index);
  }
  std::printf("]}\n");
  return 0;
}

int run_launch_overhead(const qw38::cuda::ResidentModel& model,
                        const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 1 : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphFfnOnly);

  ArmResult warmup;
  qw38::Status status = run_decode(model, tokens, prefix, outputs, &warmup);
  if (!status.is_ok()) return fail_status(status);

  std::vector<qw38::cuda::EngineOpRecord> storage(kRecordCap);
  qw38::cuda::EngineAttribution families{};
  families.records = storage.data();
  families.capacity = storage.size();
  families.record = true;
  qw38::cuda::copy_cstr(families.engine, sizeof(families.engine), "quartz");
  qw38::cuda::copy_cstr(families.phase, sizeof(families.phase), "decode");
  qw38::cuda::copy_cstr(families.graph_mode, sizeof(families.graph_mode),
                        "ffn_only");
  cudaDeviceSynchronize();
  if (qw38::cuda::record_sequence_epoch(&families, nullptr) != cudaSuccess) {
    return 1;
  }
  qw38::cuda::DecodeAttribution attribution;
  attribution.record_leaves = true;
  attribution.families = &families;
  ArmResult measured;
  status =
      run_decode(model, tokens, prefix, outputs, &measured, &attribution);
  qw38::cuda::destroy_sequence_epoch(&families);
  if (!status.is_ok()) return fail_status(status);

  std::printf("QW38_OPT114_RECORDS=[");
  for (std::size_t index = 0; index < families.count; ++index) {
    qw38::cuda::write_engine_record_json(stdout, families.records[index],
                                         index + 1 == families.count);
  }
  std::printf("]\n");
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-114\","
      "\"workload\":\"launch-overhead\",\"prefix\":%zu,\"tokens\":%zu,"
      "\"record_count\":%zu,\"pool_overflow\":%s,"
      "\"legacy_other_idle_ms\":%.9g,\"legacy_other_idle_measured\":%s,"
      "\"wall_ms\":%.9g,\"graph_ms\":%.9g,\"tok_s\":%.9g,"
      "\"launch_param_updates\":%u,\"keep\":false,"
      "\"observed_warmups\":1,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\"}\n",
      kPrefix, prefix, outputs, families.count,
      json_bool(families.pool_overflow),
      static_cast<double>(attribution.other_idle.milliseconds),
      json_bool(attribution.other_idle.measured),
      static_cast<double>(attribution.wall.milliseconds),
      static_cast<double>(attribution.graph.milliseconds),
      static_cast<double>(measured.tok_s), 0u);
  return families.pool_overflow ? 1 : 0;
}

int run_graph_ab(const qw38::cuda::ResidentModel& model, const Options& options) {
  qw38::cuda::ExecutionGraphPathScope scope(options.execution_graphs);
  const std::size_t capacity = session_capacity(options.prefix, options.tokens);
  qw38::cuda::SchedulerSession graph_session;
  qw38::cuda::SchedulerWorkspace graph_workspace;
  qw38::cuda::SchedulerSession eager_session;
  qw38::cuda::SchedulerWorkspace eager_workspace;
  qw38::Status status = graph_session.create(capacity);
  if (status.is_ok()) status = graph_workspace.create(capacity);
  if (status.is_ok()) status = eager_session.create(capacity);
  if (status.is_ok()) status = eager_workspace.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &graph_workspace);
  if (!status.is_ok()) return fail_status(status);
  const bool segments =
      std::strcmp(options.execution_graphs,
                  qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0;
  bool passed =
      (segments && graphs.decode_segment_graph_count() == 8 &&
       graphs.decode_graph_count() == 0) ||
      (!segments && graphs.decode_graph_count() == qw38::internal::kModelLayerCount &&
       graphs.decode_segment_graph_count() == 0);
  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  float graph_ms = 0.0F;
  float eager_ms = 0.0F;
  status = run_token(model, 42, &graph_session, &graph_workspace,
                     graph_logits.data(), graph_hidden.data(), &graph_ms, &graphs,
                     nullptr);
  if (status.is_ok()) {
    status = run_token(model, 42, &eager_session, &eager_workspace,
                       eager_logits.data(), eager_hidden.data(), &eager_ms,
                       nullptr, nullptr);
  }
  if (!status.is_ok()) return fail_status(status);
  passed = passed && graph_logits.size() == eager_logits.size() &&
           std::memcmp(graph_logits.data(), eager_logits.data(),
                       graph_logits.size() * sizeof(float)) == 0 &&
           std::memcmp(graph_hidden.data(), eager_hidden.data(),
                       graph_hidden.size() * sizeof(float)) == 0;
  std::printf(
      "decode_graph_dispatch path=%s decode_graph_count=%zu "
      "decode_segment_graph_count=%zu launch_param_updates=%u\n",
      graphs.execution_graph_path(), graphs.decode_graph_count(),
      graphs.decode_segment_graph_count(), graphs.launch_param_update_count());
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-114\",\"workload\":\"graph-ab\","
      "\"execution_graphs\":\"%s\",\"decode_graph_count\":%zu,"
      "\"decode_segment_graph_count\":%zu,\"launch_param_updates\":%u,"
      "\"exact_output\":%s,\"keep\":false,\"pass\":%s}\n",
      kPrefix, options.execution_graphs, graphs.decode_graph_count(),
      graphs.decode_segment_graph_count(), graphs.launch_param_update_count(),
      json_bool(passed), json_bool(passed));
  return passed ? 0 : 1;
}

int run_graph_bench(const qw38::cuda::ResidentModel& model,
                    const Options& options) {
  qw38::cuda::ExecutionGraphPathScope scope(options.execution_graphs);
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 1 : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  const std::size_t capacity = session_capacity(prefix, outputs);
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = workspace.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return fail_status(status);
  std::printf(
      "decode_graph_dispatch path=%s decode_graph_count=%zu "
      "decode_segment_graph_count=%zu launch_param_updates=%u\n",
      graphs.execution_graph_path(), graphs.decode_graph_count(),
      graphs.decode_segment_graph_count(), graphs.launch_param_update_count());
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_decode(model, tokens, prefix, outputs, &discarded);
    if (!status.is_ok()) return fail_status(status);
  }
  for (std::size_t sample = 0; sample < options.samples; ++sample) {
    ArmResult measured;
    status = run_decode(model, tokens, prefix, outputs, &measured);
    if (!status.is_ok()) return fail_status(status);
    std::printf(
        "{\"cache_mode\":\"rotating\",\"sample_index\":%zu,\"enclosing_ms\":%.9g,"
        "\"tok_s\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g}\n",
        sample, static_cast<double>(measured.wall_ms),
        static_cast<double>(measured.tok_s),
        static_cast<double>(measured.p50_ms),
        static_cast<double>(measured.p95_ms));
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-114\",\"workload\":\"graph-bench\","
      "\"execution_graphs\":\"%s\",\"prefix\":%zu,\"tokens\":%zu,"
      "\"warmups\":%zu,\"samples\":%zu,\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":1,\"observed_shapes\":1,"
      "\"observed_tier\":\"acceptance\",\"keep\":false,"
      "\"decode_graph_count\":%zu,\"decode_segment_graph_count\":%zu,"
      "\"launch_param_updates\":%u}\n",
      kPrefix, options.execution_graphs, prefix, outputs, options.warmups,
      options.samples, options.warmups, options.samples,
      graphs.decode_graph_count(), graphs.decode_segment_graph_count(),
      graphs.launch_param_update_count());
  return 0;
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

  qw38::internal::MappedFile mapping;
  qw38::cuda::ResidentModel model;
  const qw38::Status loaded =
      load_model(argv[model_index], &mapping, &model);
  if (!loaded.is_ok()) return fail_status(loaded);

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::printf("device=%s compute=%d.%d\n", prop.name, prop.major, prop.minor);

  if (std::strcmp(options.workload, "launch-overhead") == 0) {
    return run_launch_overhead(model, options);
  }
  if (std::strcmp(options.workload, "graph-ab") == 0) {
    return run_graph_ab(model, options);
  }
  if (std::strcmp(options.workload, "graph-bench") == 0) {
    return run_graph_bench(model, options);
  }
  if (std::strcmp(options.workload, "aa-control") != 0) {
    std::fprintf(stderr, "unknown workload %s\n", options.workload);
    return 2;
  }
  return run_aa_control(model, options);
}
