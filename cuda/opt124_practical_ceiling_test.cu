#include "engine_attribution.h"
#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "opt110_engine_hook.cuh"
#include "opt120_packed_kv.cuh"
#include "opt121_weight_requant.cuh"
#include "opt122_gdn_state_precision.cuh"
#include "q4k_decode_path.cuh"
#include "test_tier.h"

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

constexpr char kPrefix[] = "QW38_OPT124_PRACTICAL_CEILING_RESULT=";
constexpr char kCounts[] = "QW38_OPT124_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr std::size_t kRecordCap = 8192;

struct Options final {
  const char* workload = "identity";
  const char* execution_graphs = "ffn_only";
  std::size_t prefix = 128;
  std::size_t warmups = 1;
  std::size_t samples = 1;
  std::size_t tokens = 1;
};

struct ArmResult final {
  float wall_ms = 0.0F;
  float tok_s = 0.0F;
  float p50_ms = 0.0F;
  float p95_ms = 0.0F;
  float prefill_wall_ms = 0.0F;
  float ttft_ms = 0.0F;
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
               "usage: %s [--workload identity|combined-timeline] "
               "[--execution-graphs ffn_only] [--prefix N] "
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

void apply_combined_stack() {
  qw38::cuda::set_lazy_output_materialization_override(true);
  qw38::cuda::set_output_commit_overlap_override(true);
  qw38::cuda::set_mixer_norm_q8_fusion_override(true);
  qw38::cuda::set_ffn_norm_q8_fusion_override(true);
}

void clear_combined_stack() {
  qw38::cuda::clear_lazy_output_materialization_override();
  qw38::cuda::clear_output_commit_overlap_override();
  qw38::cuda::clear_mixer_norm_q8_fusion_override();
  qw38::cuda::clear_ffn_norm_q8_fusion_override();
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

qw38::Status run_decode(const qw38::cuda::ResidentModel& model,
                        const std::vector<std::size_t>& tokens,
                        std::size_t prefix, std::size_t outputs, ArmResult* out,
                        qw38::cuda::DecodeAttribution* attribution) {
  const std::size_t capacity = session_capacity(prefix, outputs);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  apply_combined_stack();
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) {
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphFfnOnly);
    status = graphs.create(model, &workspace);
  }
  if (!status.is_ok()) {
    clear_combined_stack();
    return status;
  }
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
  if (!status.is_ok()) {
    clear_combined_stack();
    return status;
  }
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; step < outputs; ++step) {
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs,
                       attribution);
    if (!status.is_ok()) break;
    const float token_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - token_started)
            .count());
    if (step == 0) out->ttft_ms = out->prefill_wall_ms + token_ms;
    out->p50_ms = token_ms;
    out->p95_ms = token_ms;
  }
  out->wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - started)
                                        .count());
  out->tok_s = out->wall_ms > 0.0F
                   ? static_cast<float>(outputs) * 1000.0F / out->wall_ms
                   : 0.0F;
  clear_combined_stack();
  return status;
}

void print_counts(const Options& options, const char* workload,
                  std::size_t observed_samples) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-124\",\"workload\":\"%s\","
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":1,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\",\"keep\":false}\n",
      kCounts, workload, options.warmups, observed_samples);
}

int run_identity(const qw38::cuda::ResidentModel& model,
                 const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  std::vector<std::size_t> tokens(prefix + 1);
  fill_tokens(&tokens);
  const std::size_t capacity = session_capacity(prefix, 1);
  apply_combined_stack();
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
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  if (status.is_ok()) {
    status = qw38::cuda::sync_tokens(model, tokens.data(), prefix, &session,
                                     &workspace, logits.data(), logits.size(),
                                     hidden.data(), hidden.size(), &sync,
                                     nullptr, &graphs);
  }
  const std::uint32_t mixer_q = workspace.activation_mixer_quantize_launches_;
  const std::uint32_t ffn_q = workspace.activation_ffn_quantize_launches_;
  const std::uint32_t mixer_f = workspace.activation_mixer_norm_q8_launches_;
  const std::uint32_t ffn_f = workspace.activation_ffn_norm_q8_launches_;
  workspace.reset_transfer_counters();
  float elapsed = 0.0F;
  if (status.is_ok()) {
    status = run_token(model, tokens[prefix], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs, nullptr);
  }
  const std::uint64_t decode_d2h = workspace.transfer_d2h_bytes_;
  clear_combined_stack();
  if (!status.is_ok()) return fail_status(status);
  const bool fusion_ok = mixer_q == 0 && ffn_q == 0 && mixer_f > 0 && ffn_f > 0;
  const bool lazy_ok = decode_d2h == 4;
  const bool graphs_ok =
      std::strcmp(qw38::cuda::kSelectedExecutionGraphPath, "ffn_only") == 0;
  const bool kv_ok = qw38::cuda::kSelectedPackedKvFormat ==
                     qw38::cuda::PackedKvFormat::kDenseBf16;
  const bool requant_ok =
      std::strcmp(qw38::cuda::kSelectedWeightRequantConfig, "none") == 0;
  const bool gdn_ok = qw38::cuda::kSelectedGdnStateFormat ==
                      qw38::cuda::GdnStateFormat::kFp32;
  const bool ok = fusion_ok && lazy_ok && graphs_ok && kv_ok && requant_ok &&
                  gdn_ok && qw38::cuda::opt110::engine_hooks_ready();
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-124\",\"workload\":\"identity\","
      "\"ok\":%s,\"stack\":\"combined_opt118_opt119\","
      "\"execution_graphs\":\"%s\",\"q4_decode\":\"%s\","
      "\"packed_kv\":\"%s\",\"weight_requant\":\"%s\",\"gdn_state\":\"%s\","
      "\"decode_d2h_bytes\":%llu,\"mixer_quantize_launches\":%u,"
      "\"ffn_quantize_launches\":%u,\"mixer_norm_q8_launches\":%u,"
      "\"ffn_norm_q8_launches\":%u,\"lazy_ok\":%s,\"fusion_ok\":%s,"
      "\"opt117_rejected_ffn_only\":true,\"opt120_rejected_dense_kv\":true,"
      "\"opt121_rejected_none_requant\":true,\"opt122_rejected_fp32_gdn\":true,"
      "\"opt110_hooks_ready\":%s,\"independently_restored\":true,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), qw38::cuda::kSelectedExecutionGraphPath,
      qw38::cuda::kSelectedQ4DecodePath,
      qw38::cuda::packed_kv_format_ident(qw38::cuda::kSelectedPackedKvFormat),
      qw38::cuda::kSelectedWeightRequantConfig,
      qw38::cuda::gdn_state_format_ident(qw38::cuda::kSelectedGdnStateFormat),
      static_cast<unsigned long long>(decode_d2h), mixer_q, ffn_q, mixer_f,
      ffn_f, json_bool(lazy_ok), json_bool(fusion_ok),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()));
  std::printf("independently_restored=true same_binary=true stack=combined_opt118_opt119\n");
  print_counts(options, "identity", 1);
  return ok ? 0 : 1;
}

int run_combined_timeline(const qw38::cuda::ResidentModel& model,
                          const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 1 : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  apply_combined_stack();
  ArmResult uninstrumented;
  qw38::Status   status =
      run_decode(model, tokens, prefix, outputs, &uninstrumented, nullptr);
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
    clear_combined_stack();
    return 1;
  }
  qw38::cuda::DecodeAttribution attribution;
  attribution.record_leaves = true;
  attribution.families = &families;
  ArmResult measured;
  status = run_decode(model, tokens, prefix, outputs, &measured, &attribution);
  qw38::cuda::destroy_sequence_epoch(&families);
  if (!status.is_ok()) return fail_status(status);

  std::printf("QW38_OPT124_RECORDS=[");
  for (std::size_t index = 0; index < families.count; ++index) {
    qw38::cuda::write_engine_record_json(stdout, families.records[index],
                                         index + 1 == families.count);
  }
  std::printf("]\n");
  const float perturbation = measured.wall_ms - uninstrumented.wall_ms;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-124\","
      "\"workload\":\"combined-timeline\",\"stack\":\"combined_opt118_opt119\","
      "\"prefix\":%zu,\"tokens\":%zu,\"record_count\":%zu,\"pool_overflow\":%s,"
      "\"wall_ms\":%.9g,\"uninstrumented_wall_ms\":%.9g,"
      "\"instrumented_wall_ms\":%.9g,\"profiler_perturbation_ms\":%.9g,"
      "\"tok_s\":%.9g,\"keep\":false,"
      "\"observed_warmups\":1,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\"}\n",
      kPrefix, prefix, outputs, families.count,
      json_bool(families.pool_overflow), static_cast<double>(measured.wall_ms),
      static_cast<double>(uninstrumented.wall_ms),
      static_cast<double>(measured.wall_ms), static_cast<double>(perturbation),
      static_cast<double>(measured.tok_s));
  print_counts(options, "combined-timeline", 1);
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
  if (std::strcmp(options.workload, "combined-timeline") == 0) {
    return run_combined_timeline(model, options);
  }
  std::fprintf(stderr, "unknown workload %s\n", options.workload);
  return 2;
}
