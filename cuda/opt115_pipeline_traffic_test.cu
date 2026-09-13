#include "engine_attribution.h"
#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "opt110_engine_hook.cuh"
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

constexpr char kPrefix[] = "QW38_OPT115_PIPELINE_TRAFFIC_RESULT=";
constexpr char kCounts[] = "QW38_OPT115_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr std::size_t kCapacity128k = 131072;
constexpr std::size_t kRecordCap = 8192;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;
constexpr std::size_t kGdnLayers = 48;
constexpr std::size_t kAttentionLayers = 16;

struct Options final {
  const char* workload = "request-trace";
  const char* execution_graphs = "ffn_only";
  std::size_t prefix = 128;
  std::size_t capacity = 0;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  std::fprintf(stderr, "opt110_engine_hooks_ready=%s llama_q81_path=post113\n",
               json_bool(qw38::cuda::opt110::engine_hooks_ready()));
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload request-trace|session-loop|long-context|traffic-timeline] "
               "[--execution-graphs ffn_only] [--prefix N] [--capacity N] "
               "[--warmups N] [--samples N] [--tokens N] MODEL\n",
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
    } else if (std::strcmp(arg, "--capacity") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->capacity)) return usage(argv[0]);
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

std::size_t session_capacity(std::size_t prefix, std::size_t outputs,
                             std::size_t capacity_override) {
  const std::size_t needed = prefix + outputs;
  if (capacity_override != 0) return capacity_override;
  return std::max(needed + 32, kMinCapacity);
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

std::size_t kv_append_bytes() {
  return kAttentionLayers * qw38::internal::kAttentionKvWidth * 2U *
         sizeof(__nv_bfloat16);
}

std::size_t gdn_state_bytes() {
  return kGdnLayers *
         (qw38::internal::kGdnConvolutionValues +
          qw38::internal::kGdnRecurrentStateValues) *
         sizeof(float);
}

std::size_t logits_d2h_bytes() {
  return qw38::internal::kVocabularySize * sizeof(float);
}

std::size_t hidden_d2h_bytes() {
  return qw38::internal::kResidualWidth * sizeof(float);
}

std::size_t greedy_sample(const float* logits) {
  std::size_t best = 0;
  float value = logits[0];
  for (std::size_t index = 1; index < qw38::internal::kVocabularySize; ++index) {
    if (logits[index] > value) {
      value = logits[index];
      best = index;
    }
  }
  return best;
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
  float prefill_wall_ms = 0.0F;
  float ttft_ms = 0.0F;
  std::size_t sampled_token = 0;
  std::size_t host_sample_ms_x1000 = 0;
};

qw38::Status run_prefill(const qw38::cuda::ResidentModel& model,
                         const std::vector<std::size_t>& tokens,
                         std::size_t prompt, ArmResult* out,
                         std::size_t capacity_override = 0) {
  const std::size_t capacity = session_capacity(prompt, 0, capacity_override);
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
  out->prefill_wall_ms = out->wall_ms;
  out->ttft_ms = out->wall_ms;
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
                        qw38::cuda::DecodeAttribution* attribution = nullptr,
                        std::size_t capacity_override = 0) {
  const std::size_t capacity =
      session_capacity(prefix, outputs, capacity_override);
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
  const auto prefill_started = std::chrono::steady_clock::now();
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), prefix, &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &sync, nullptr, &graphs);
  out->prefill_wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - prefill_started)
          .count());
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
    const float token_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - token_started)
            .count());
    latencies.push_back(token_ms);
    if (step == 0) out->ttft_ms = out->prefill_wall_ms + token_ms;
  }
  out->wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - started)
                                        .count());
  out->tok_s = out->wall_ms > 0.0F
                   ? static_cast<float>(outputs) * 1000.0F / out->wall_ms
                   : 0.0F;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  if (outputs == 0) {
    out->ttft_ms = out->prefill_wall_ms;
    out->p50_ms = out->wall_ms;
    out->p95_ms = out->wall_ms;
  }
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

void print_counts(const Options& options, const char* workload,
                  std::size_t observed_samples) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-115\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":2,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":false,"
      "\"workload\":\"%s\",\"sample_ids\":[",
      kCounts, options.warmups, observed_samples, workload);
  for (std::size_t index = 0; index < observed_samples; ++index) {
    if (index != 0) std::printf(",");
    std::printf("%zu", index);
  }
  std::printf("]}\n");
}

int run_request_trace(const qw38::cuda::ResidentModel& model,
                      const Options& options) {
  const bool prefill = options.prefix == 4096 || options.tokens == 0;
  const std::size_t prompt = prefill ? options.prefix : options.prefix;
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
      "%s{\"schema_version\":1,\"task\":\"OPT-115\","
      "\"workload\":\"request-trace\","
      "\"prefix\":%zu,\"prefill\":%s,\"warmups\":%zu,\"samples\":%zu,"
      "\"tokens\":%zu,\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\",\"keep\":false,"
      "\"execution_graphs\":\"%s\",\"same_binary\":true,"
      "\"independently_restored\":true,\"opt110_hooks_ready\":%s,\"pairs\":[",
      kPrefix, options.prefix, json_bool(prefill), options.warmups,
      options.samples, outputs, options.warmups, options.samples,
      qw38::cuda::effective_execution_graph_path(),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()));
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    if (pair != 0) std::printf(",");
    std::printf("{\"sample_index\":%zu,\"order\":\"%s\",", pair,
                (pair % 2) == 1 ? "BA" : "AB");
    print_arm("A", arm_a[pair], false);
    print_arm("B", arm_b[pair], true);
    std::printf("}");
  }
  std::printf("]}\n");
  print_counts(options, "request-trace", options.samples);
  return 0;
}

int run_session_loop(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 4096 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 32 : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  qw38::cuda::ExecutionGraphPathScope scope(options.execution_graphs);
  const std::size_t capacity =
      session_capacity(prefix, outputs, options.capacity);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  const auto request_started = std::chrono::steady_clock::now();
  const auto prefill_started = request_started;
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), prefix, &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &sync, nullptr, &graphs);
  const float prefill_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - prefill_started)
          .count());
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> latencies;
  latencies.reserve(outputs);
  float ttft_ms = prefill_ms;
  double host_sample_ms = 0.0;
  std::size_t last_sampled = 0;
  for (std::size_t step = 0; step < outputs; ++step) {
    const auto sample_started = std::chrono::steady_clock::now();
    last_sampled = greedy_sample(logits.data());
    host_sample_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - sample_started)
                          .count();
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs, nullptr);
    if (!status.is_ok()) return fail_status(status);
    const float token_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - token_started)
            .count());
    latencies.push_back(token_ms);
    if (step == 0) ttft_ms = prefill_ms + token_ms;
  }
  const float wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - request_started)
          .count());
  const float decode_ms = wall_ms - prefill_ms;
  const float tok_s =
      decode_ms > 0.0F ? static_cast<float>(outputs) * 1000.0F / decode_ms : 0.0F;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-115\","
      "\"workload\":\"session-loop\",\"prefix\":%zu,\"tokens\":%zu,"
      "\"capacity\":%zu,\"wall_ms\":%.9g,\"prefill_wall_ms\":%.9g,"
      "\"ttft_ms\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,"
      "\"host_sample_ms\":%.9g,\"last_sampled_token\":%zu,"
      "\"planned_tokens\":true,\"session_sync_sample_eval\":true,"
      "\"keep\":false,\"status\":\"measured\"}\n",
      kPrefix, prefix, outputs, capacity, static_cast<double>(wall_ms),
      static_cast<double>(prefill_ms), static_cast<double>(ttft_ms),
      static_cast<double>(tok_s), static_cast<double>(percentile(latencies, 0.50F)),
      static_cast<double>(percentile(latencies, 0.95F)), host_sample_ms,
      last_sampled);
  print_counts(options, "session-loop", 1);
  return 0;
}

int run_long_context(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
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
  qw38::cuda::ExecutionGraphPathScope scope(options.execution_graphs);
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_decode(model, tokens, prefix, outputs, &discarded, nullptr,
                        capacity);
    if (!status.is_ok()) return fail_status(status);
    std::printf("long_warmup=%zu tok_s=%.9g wall_ms=%.9g populated_cache=true\n",
                warmup, static_cast<double>(discarded.tok_s),
                static_cast<double>(discarded.wall_ms));
  }
  ArmResult measured;
  status =
      run_decode(model, tokens, prefix, outputs, &measured, nullptr, capacity);
  if (!status.is_ok()) return fail_status(status);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-115\","
      "\"workload\":\"long-context\",\"prefix\":%zu,\"tokens\":%zu,"
      "\"capacity\":%zu,\"wall_ms\":%.9g,\"prefill_wall_ms\":%.9g,"
      "\"ttft_ms\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,"
      "\"populated_cache\":true,\"capacity_alone_is_not_traffic\":true,"
      "\"keep\":false,\"status\":\"measured\"}\n",
      kPrefix, prefix, outputs, capacity, static_cast<double>(measured.wall_ms),
      static_cast<double>(measured.prefill_wall_ms),
      static_cast<double>(measured.ttft_ms), static_cast<double>(measured.tok_s),
      static_cast<double>(measured.p50_ms), static_cast<double>(measured.p95_ms));
  print_counts(options, "long-context", options.samples == 0 ? 1 : options.samples);
  return 0;
}

int run_traffic_timeline(const qw38::cuda::ResidentModel& model,
                         const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 1 : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphFfnOnly);

  ArmResult uninstrumented;
  qw38::Status status =
      run_decode(model, tokens, prefix, outputs, &uninstrumented);
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
  status = run_decode(model, tokens, prefix, outputs, &measured, &attribution);
  qw38::cuda::destroy_sequence_epoch(&families);
  if (!status.is_ok()) return fail_status(status);

  std::printf("QW38_OPT115_RECORDS=[");
  for (std::size_t index = 0; index < families.count; ++index) {
    qw38::cuda::write_engine_record_json(stdout, families.records[index],
                                         index + 1 == families.count);
  }
  std::printf("]\n");
  const float perturbation =
      measured.wall_ms - uninstrumented.wall_ms;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-115\","
      "\"workload\":\"traffic-timeline\",\"prefix\":%zu,\"tokens\":%zu,"
      "\"record_count\":%zu,\"pool_overflow\":%s,"
      "\"wall_ms\":%.9g,\"uninstrumented_wall_ms\":%.9g,"
      "\"instrumented_wall_ms\":%.9g,\"profiler_perturbation_ms\":%.9g,"
      "\"tok_s\":%.9g,\"logits_d2h_bytes\":%zu,\"hidden_d2h_bytes\":%zu,"
      "\"kv_append_bytes\":%zu,\"gdn_state_bytes\":%zu,"
      "\"kv_read_bytes\":%zu,\"keep\":false,"
      "\"observed_warmups\":1,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\"}\n",
      kPrefix, prefix, outputs, families.count,
      json_bool(families.pool_overflow), static_cast<double>(measured.wall_ms),
      static_cast<double>(uninstrumented.wall_ms),
      static_cast<double>(measured.wall_ms), static_cast<double>(perturbation),
      static_cast<double>(measured.tok_s), logits_d2h_bytes(),
      hidden_d2h_bytes(), kv_append_bytes(), gdn_state_bytes(),
      kAttentionLayers * prefix * qw38::internal::kAttentionKvWidth * 2U *
          sizeof(__nv_bfloat16));
  print_counts(options, "traffic-timeline", 1);
  return families.pool_overflow ? 1 : 0;
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
  std::printf("device=%s compute=%d.%d opt110_engine_hooks_ready=%s\n", prop.name,
              prop.major, prop.minor,
              json_bool(qw38::cuda::opt110::engine_hooks_ready()));

  if (std::strcmp(options.workload, "session-loop") == 0) {
    return run_session_loop(model, options);
  }
  if (std::strcmp(options.workload, "long-context") == 0) {
    return run_long_context(model, options);
  }
  if (std::strcmp(options.workload, "traffic-timeline") == 0) {
    return run_traffic_timeline(model, options);
  }
  if (std::strcmp(options.workload, "request-trace") != 0) {
    std::fprintf(stderr, "unknown workload %s\n", options.workload);
    return 2;
  }
  return run_request_trace(model, options);
}
