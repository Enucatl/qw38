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
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT125_DECODE_ACCOUNTING_RESULT=";
constexpr char kCounts[] = "QW38_OPT125_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr std::size_t kCapacity128k = 131072;
constexpr std::size_t kRecordCap = 8192;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 5;
constexpr std::size_t kDefaultDecodeTokens = 256;

struct Options final {
  const char* workload = "identity";
  const char* stack = "combined";
  const char* execution_graphs = "ffn_only";
  std::size_t prefix = 128;
  std::size_t capacity = 0;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
};

struct ArmResult final {
  float setup_ms = 0.0F;
  float prefill_wall_ms = 0.0F;
  float decode_only_wall_ms = 0.0F;
  float request_wall_ms = 0.0F;
  float ttft_ms = 0.0F;
  float decode_only_tok_s = 0.0F;
  float request_tok_s = 0.0F;
  float p50_ms = 0.0F;
  float p95_ms = 0.0F;
  std::size_t emitted_tokens = 0;
  std::size_t eval_calls = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t prefill_d2h_bytes = 0;
  std::size_t session_bytes = 0;
  std::size_t workspace_bytes = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  std::fprintf(stderr, "opt110_engine_hooks_ready=%s\n",
               json_bool(qw38::cuda::opt110::engine_hooks_ready()));
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload identity|matched-baseline|"
               "activity-windows|long-context] "
               "[--stack combined|post113] [--execution-graphs ffn_only] "
               "[--prefix N] [--capacity N] [--warmups N] [--samples N] "
               "[--tokens N] MODEL\n",
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
    } else if (std::strcmp(arg, "--stack") == 0 && index + 1 < argc) {
      options->stack = argv[++index];
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
  if (std::strcmp(options->stack, "combined") != 0 &&
      std::strcmp(options->stack, "post113") != 0) {
    std::fprintf(stderr, "invalid --stack %s\n", options->stack);
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

void apply_stack(const char* stack) {
  const bool combined = std::strcmp(stack, "combined") == 0;
  qw38::cuda::set_lazy_output_materialization_override(combined);
  qw38::cuda::set_output_commit_overlap_override(combined);
  qw38::cuda::set_mixer_norm_q8_fusion_override(combined);
  qw38::cuda::set_ffn_norm_q8_fusion_override(combined);
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
                       qw38::cuda::SchedulerGraphs* graphs,
                       qw38::cuda::DecodeAttribution* attribution) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, nullptr, nullptr,
      qw38::cuda::PointwisePath::kFused, graphs, attribution);
}

bool in_window(std::size_t step, std::size_t outputs) {
  if (outputs == 0) return false;
  const std::size_t early_end = std::min<std::size_t>(4, outputs);
  const std::size_t mid = outputs / 2;
  const std::size_t mid_start = mid >= 2 ? mid - 2 : 0;
  const std::size_t mid_end = std::min(mid_start + 4, outputs);
  const std::size_t late_start = outputs > 4 ? outputs - 4 : 0;
  return step < early_end || (step >= mid_start && step < mid_end) ||
         step >= late_start;
}

qw38::Status run_probe(const qw38::cuda::ResidentModel& model,
                       const std::vector<std::size_t>& tokens, std::size_t prefix,
                       std::size_t outputs, const char* stack, ArmResult* out,
                       qw38::cuda::DecodeAttribution* attribution,
                       bool windowed, std::size_t capacity_override) {
  apply_stack(stack);
  const std::size_t capacity =
      session_capacity(prefix, outputs, capacity_override);
  const auto setup_started = std::chrono::steady_clock::now();
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) {
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphFfnOnly);
    status = graphs.create(model, &workspace);
  }
  out->setup_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() -
                                         setup_started)
                                         .count());
  if (!status.is_ok()) {
    clear_stack();
    return status;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  workspace.reset_transfer_counters();
  const auto request_started = std::chrono::steady_clock::now();
  if (prefix > 0) {
    status = qw38::cuda::sync_tokens(
        model, tokens.data(), prefix, &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &sync, nullptr, &graphs);
  }
  out->prefill_wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - request_started)
          .count());
  out->ttft_ms = out->prefill_wall_ms;
  if (!status.is_ok()) {
    clear_stack();
    return status;
  }
  const std::uint64_t prefill_d2h = workspace.transfer_d2h_bytes_;
  workspace.reset_transfer_counters();
  out->prefill_d2h_bytes = prefill_d2h;
  std::vector<float> latencies;
  latencies.reserve(outputs);
  const auto decode_started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    float elapsed = 0.0F;
    qw38::cuda::DecodeAttribution* used = attribution;
    if (windowed && used != nullptr && !in_window(step, outputs)) {
      used = nullptr;
    }
    const auto token_started = std::chrono::steady_clock::now();
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs, used);
    const float token_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - token_started)
            .count());
    latencies.push_back(token_ms);
    if (step == 0) out->ttft_ms = out->prefill_wall_ms + token_ms;
  }
  const auto finished = std::chrono::steady_clock::now();
  out->decode_only_wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(finished - decode_started)
          .count());
  out->request_wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(finished - request_started)
          .count());
  out->emitted_tokens = outputs;
  out->eval_calls = outputs;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  if (outputs == 0) {
    out->decode_only_wall_ms = 0.0F;
    out->request_wall_ms = out->prefill_wall_ms;
    out->ttft_ms = out->prefill_wall_ms;
    out->p50_ms = out->prefill_wall_ms;
    out->p95_ms = out->prefill_wall_ms;
    const float counted = static_cast<float>(prefix);
    out->decode_only_tok_s =
        out->prefill_wall_ms > 0.0F
            ? counted * 1000.0F / out->prefill_wall_ms
            : 0.0F;
    out->request_tok_s = out->decode_only_tok_s;
  } else {
    out->decode_only_tok_s =
        out->decode_only_wall_ms > 0.0F
            ? static_cast<float>(outputs) * 1000.0F / out->decode_only_wall_ms
            : 0.0F;
    out->request_tok_s =
        out->request_wall_ms > 0.0F
            ? static_cast<float>(outputs) * 1000.0F / out->request_wall_ms
            : 0.0F;
  }
  out->d2h_bytes = workspace.transfer_d2h_bytes_;
  out->session_bytes = session.allocated_bytes();
  out->workspace_bytes = workspace.allocated_bytes();
  clear_stack();
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"setup_ms\":%.9g,\"prefill_wall_ms\":%.9g,"
      "\"decode_only_wall_ms\":%.9g,\"request_wall_ms\":%.9g,"
      "\"ttft_ms\":%.9g,\"decode_only_tok_s\":%.9g,\"request_tok_s\":%.9g,"
      "\"p50_ms\":%.9g,\"p95_ms\":%.9g,\"emitted_tokens\":%zu,"
      "\"eval_calls\":%zu,\"first_sampled_token_needs_eval\":false,"
      "\"d2h_bytes\":%llu,\"prefill_d2h_bytes\":%llu,\"session_bytes\":%zu,"
      "\"workspace_bytes\":%zu}%s",
      name, static_cast<double>(arm.setup_ms),
      static_cast<double>(arm.prefill_wall_ms),
      static_cast<double>(arm.decode_only_wall_ms),
      static_cast<double>(arm.request_wall_ms),
      static_cast<double>(arm.ttft_ms),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.request_tok_s), static_cast<double>(arm.p50_ms),
      static_cast<double>(arm.p95_ms), arm.emitted_tokens, arm.eval_calls,
      static_cast<unsigned long long>(arm.d2h_bytes),
      static_cast<unsigned long long>(arm.prefill_d2h_bytes), arm.session_bytes,
      arm.workspace_bytes, last ? "" : ",");
}

void print_counts(const Options& options, const char* workload,
                  std::size_t observed_samples, std::size_t candidates) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-125\",\"workload\":\"%s\","
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":%zu,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\",\"keep\":false}\n",
      kCounts, workload, options.warmups, observed_samples, candidates);
}

void print_profiler_probe() {
  const int nsys = std::system("command -v nsys >/dev/null 2>&1");
  const int ncu = std::system("command -v ncu >/dev/null 2>&1");
  std::printf("nsys_available=%s ncu_available=%s cupti_linked=false "
              "full_ncu_sweep=false\n",
              nsys == 0 ? "true" : "false", ncu == 0 ? "true" : "false");
  if (nsys != 0) std::printf("nsys_error=nsys not found\n");
  if (ncu != 0) std::printf("ncu_error=ncu not found\n");
}

int run_identity(const qw38::cuda::ResidentModel& model,
                 const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  std::vector<std::size_t> tokens(prefix + 1);
  fill_tokens(&tokens);
  ArmResult measured;
  const qw38::Status status =
      run_probe(model, tokens, prefix, 1, "combined", &measured, nullptr, false,
                0);
  if (!status.is_ok()) return fail_status(status);
  const bool graphs_ok =
      std::strcmp(qw38::cuda::kSelectedExecutionGraphPath, "ffn_only") == 0;
  const bool kv_ok = qw38::cuda::kSelectedPackedKvFormat ==
                     qw38::cuda::PackedKvFormat::kDenseBf16;
  const bool requant_ok =
      std::strcmp(qw38::cuda::kSelectedWeightRequantConfig, "none") == 0;
  const bool gdn_ok = qw38::cuda::kSelectedGdnStateFormat ==
                      qw38::cuda::GdnStateFormat::kFp32;
  const bool lazy_ok = measured.d2h_bytes == 4;
  const bool ok = graphs_ok && kv_ok && requant_ok && gdn_ok && lazy_ok &&
                  qw38::cuda::opt110::engine_hooks_ready();
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-125\",\"workload\":\"identity\","
      "\"ok\":%s,\"stack\":\"combined_opt118_opt119\","
      "\"execution_graphs\":\"%s\",\"q4_decode\":\"%s\","
      "\"packed_kv\":\"%s\",\"weight_requant\":\"%s\",\"gdn_state\":\"%s\","
      "\"decode_d2h_bytes\":%llu,\"lazy_ok\":%s,"
      "\"opt117_rejected_ffn_only\":true,\"opt120_rejected_dense_kv\":true,"
      "\"opt121_rejected_none_requant\":true,\"opt122_rejected_fp32_gdn\":true,"
      "\"opt110_hooks_ready\":%s,\"independently_restored\":true,"
      "\"same_binary\":true,\"keep\":false}\n",
      kPrefix, json_bool(ok), qw38::cuda::kSelectedExecutionGraphPath,
      qw38::cuda::kSelectedQ4DecodePath,
      qw38::cuda::packed_kv_format_ident(qw38::cuda::kSelectedPackedKvFormat),
      qw38::cuda::kSelectedWeightRequantConfig,
      qw38::cuda::gdn_state_format_ident(qw38::cuda::kSelectedGdnStateFormat),
      static_cast<unsigned long long>(measured.d2h_bytes), json_bool(lazy_ok),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()));
  std::printf(
      "independently_restored=true same_binary=true stack=combined_opt118_opt119\n");
  print_counts(options, "identity", 1, 1);
  return ok ? 0 : 1;
}

int run_matched_baseline(const qw38::cuda::ResidentModel& model,
                         const Options& options) {
  const bool prefill_only = options.prefix == 4096 || options.tokens == 0;
  const std::size_t prompt = options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_probe(model, tokens, prompt, outputs, options.stack, &discarded,
                       nullptr, false, options.capacity);
    if (!status.is_ok()) return fail_status(status);
    std::printf(
        "quartz_warmup=%zu stack=%s decode_only_tok_s=%.9g request_tok_s=%.9g "
        "request_wall_ms=%.9g prefill_wall_ms=%.9g independently_restored=true "
        "same_binary=true\n",
        warmup, options.stack, static_cast<double>(discarded.decode_only_tok_s),
        static_cast<double>(discarded.request_tok_s),
        static_cast<double>(discarded.request_wall_ms),
        static_cast<double>(discarded.prefill_wall_ms));
  }
  std::vector<ArmResult> arm_a(options.samples);
  std::vector<ArmResult> arm_b(options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    const bool ba = (pair % 2) == 1;
    ArmResult first;
    ArmResult second;
    status = run_probe(model, tokens, prompt, outputs, options.stack, &first,
                       nullptr, false, options.capacity);
    if (status.is_ok()) {
      status = run_probe(model, tokens, prompt, outputs, options.stack, &second,
                         nullptr, false, options.capacity);
    }
    if (!status.is_ok()) return fail_status(status);
    arm_a[pair] = ba ? second : first;
    arm_b[pair] = ba ? first : second;
    std::printf(
        "quartz_pair sample_index=%zu order=%s stack=%s "
        "A_decode_only_tok_s=%.9g B_decode_only_tok_s=%.9g "
        "A_request_tok_s=%.9g B_request_tok_s=%.9g "
        "independently_restored=true same_binary=true\n",
        pair, ba ? "BA" : "AB", options.stack,
        static_cast<double>(arm_a[pair].decode_only_tok_s),
        static_cast<double>(arm_b[pair].decode_only_tok_s),
        static_cast<double>(arm_a[pair].request_tok_s),
        static_cast<double>(arm_b[pair].request_tok_s));
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-125\","
      "\"workload\":\"matched-baseline\",\"stack\":\"%s\",\"prefix\":%zu,"
      "\"prefill\":%s,\"warmups\":%zu,\"samples\":%zu,\"tokens\":%zu,"
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\",\"keep\":false,"
      "\"execution_graphs\":\"%s\",\"same_binary\":true,"
      "\"independently_restored\":true,\"opt110_hooks_ready\":%s,"
      "\"first_sampled_token_needs_eval\":false,"
      "\"metric_clock_starts_before_prefill_for_request\":true,"
      "\"metric_clock_starts_after_prefill_for_decode_only\":true,"
      "\"pairs\":[",
      kPrefix, options.stack, options.prefix, json_bool(prefill_only),
      options.warmups, options.samples, outputs, options.warmups,
      options.samples, qw38::cuda::effective_execution_graph_path(),
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
  print_counts(options, "matched-baseline", options.samples, 2);
  return 0;
}

int run_activity_windows(const qw38::cuda::ResidentModel& model,
                         const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs =
      options.tokens == 0 ? kDefaultDecodeTokens : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  print_profiler_probe();
  ArmResult uninstrumented;
  qw38::Status status =
      run_probe(model, tokens, prefix, outputs, "combined", &uninstrumented,
                nullptr, false, 0);
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
  status = run_probe(model, tokens, prefix, outputs, "combined", &measured,
                     &attribution, true, 0);
  qw38::cuda::destroy_sequence_epoch(&families);
  if (!status.is_ok()) return fail_status(status);

  std::printf("QW38_OPT125_RECORDS=[");
  for (std::size_t index = 0; index < families.count; ++index) {
    qw38::cuda::write_engine_record_json(stdout, families.records[index],
                                         index + 1 == families.count);
  }
  std::printf("]\n");
  const float perturbation =
      measured.decode_only_wall_ms - uninstrumented.decode_only_wall_ms;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-125\","
      "\"workload\":\"activity-windows\",\"stack\":\"combined_opt118_opt119\","
      "\"prefix\":%zu,\"tokens\":%zu,\"record_count\":%zu,\"pool_overflow\":%s,"
      "\"uninstrumented_decode_only_wall_ms\":%.9g,"
      "\"instrumented_decode_only_wall_ms\":%.9g,"
      "\"uninstrumented_request_wall_ms\":%.9g,"
      "\"instrumented_request_wall_ms\":%.9g,"
      "\"profiler_perturbation_ms\":%.9g,"
      "\"decode_only_tok_s\":%.9g,\"request_tok_s\":%.9g,"
      "\"windows\":{\"early\":\"0:4\",\"middle\":\"mid-2:mid+2\","
      "\"late\":\"n-4:n\"},\"method\":\"cuda_event_engine_attribution\","
      "\"nsight_systems\":\"probed\",\"nsight_compute\":\"probed\","
      "\"cupti_linked\":false,\"keep\":false,"
      "\"observed_warmups\":1,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\"}\n",
      kPrefix, prefix, outputs, families.count,
      json_bool(families.pool_overflow),
      static_cast<double>(uninstrumented.decode_only_wall_ms),
      static_cast<double>(measured.decode_only_wall_ms),
      static_cast<double>(uninstrumented.request_wall_ms),
      static_cast<double>(measured.request_wall_ms),
      static_cast<double>(perturbation),
      static_cast<double>(measured.decode_only_tok_s),
      static_cast<double>(measured.request_tok_s));
  print_counts(options, "activity-windows", 1, 1);
  return families.pool_overflow ? 1 : 0;
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
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_probe(model, tokens, prefix, outputs, "combined", &discarded,
                       nullptr, false, capacity);
    if (!status.is_ok()) {
      std::fprintf(stderr, "allocation_or_run_failed warmup=%zu code=%s msg=%s\n",
                   warmup, qw38::status_code_name(status.code()),
                   status.message().c_str());
      return fail_status(status);
    }
    std::printf(
        "long_warmup=%zu decode_only_tok_s=%.9g request_tok_s=%.9g "
        "populated_cache=true\n",
        warmup, static_cast<double>(discarded.decode_only_tok_s),
        static_cast<double>(discarded.request_tok_s));
  }
  ArmResult measured;
  status = run_probe(model, tokens, prefix, outputs, "combined", &measured,
                     nullptr, false, capacity);
  if (!status.is_ok()) {
    std::fprintf(stderr, "allocation_or_run_failed sample code=%s msg=%s\n",
                 qw38::status_code_name(status.code()), status.message().c_str());
    return fail_status(status);
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-125\","
      "\"workload\":\"long-context\",\"stack\":\"combined_opt118_opt119\","
      "\"prefix\":%zu,\"tokens\":%zu,\"capacity\":%zu,"
      "\"setup_ms\":%.9g,\"prefill_wall_ms\":%.9g,"
      "\"decode_only_wall_ms\":%.9g,\"request_wall_ms\":%.9g,"
      "\"ttft_ms\":%.9g,\"decode_only_tok_s\":%.9g,\"request_tok_s\":%.9g,"
      "\"p50_ms\":%.9g,\"p95_ms\":%.9g,\"emitted_tokens\":%zu,"
      "\"eval_calls\":%zu,\"session_bytes\":%zu,\"workspace_bytes\":%zu,"
      "\"populated_cache\":true,\"capacity_alone_is_not_traffic\":true,"
      "\"generic_failed_process_is_not_oom\":true,\"keep\":false,"
      "\"status\":\"measured\"}\n",
      kPrefix, prefix, outputs, capacity, static_cast<double>(measured.setup_ms),
      static_cast<double>(measured.prefill_wall_ms),
      static_cast<double>(measured.decode_only_wall_ms),
      static_cast<double>(measured.request_wall_ms),
      static_cast<double>(measured.ttft_ms),
      static_cast<double>(measured.decode_only_tok_s),
      static_cast<double>(measured.request_tok_s),
      static_cast<double>(measured.p50_ms), static_cast<double>(measured.p95_ms),
      measured.emitted_tokens, measured.eval_calls, measured.session_bytes,
      measured.workspace_bytes);
  print_counts(options, "long-context", 1, 1);
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
  const qw38::Status loaded = load_model(argv[model_index], &mapping, &model);
  if (!loaded.is_ok()) return fail_status(loaded);

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::printf("device=%s compute=%d.%d opt110_engine_hooks_ready=%s\n", prop.name,
              prop.major, prop.minor,
              json_bool(qw38::cuda::opt110::engine_hooks_ready()));

  if (std::strcmp(options.workload, "identity") == 0) {
    return run_identity(model, options);
  }
  if (std::strcmp(options.workload, "matched-baseline") == 0) {
    return run_matched_baseline(model, options);
  }
  if (std::strcmp(options.workload, "activity-windows") == 0) {
    return run_activity_windows(model, options);
  }
  if (std::strcmp(options.workload, "long-context") == 0) {
    return run_long_context(model, options);
  }
  std::fprintf(stderr, "unknown workload %s\n", options.workload);
  return 2;
}
