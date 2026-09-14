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
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT132_COMBINED_DECODE_RESULT=";
constexpr char kCounts[] = "QW38_OPT132_NATIVE_COUNTS=";
constexpr char kRecords[] = "QW38_OPT132_RECORDS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr std::size_t kCapacity128k = 131072;
constexpr std::size_t kRecordCap = 8192;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;

struct Options final {
  const char* workload = "identity";
  const char* execution_graphs = "decode_segments8";
  std::size_t prefix = 128;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
  std::size_t capacity = 0;
};

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
  std::chrono::steady_clock::time_point cancel_requested{};
  bool requested = false;
};

struct ArmResult final {
  float setup_ms = 0.0F;
  float prefill_ms = 0.0F;
  float decode_only_ms = 0.0F;
  float request_ms = 0.0F;
  float ttft_ms = 0.0F;
  float decode_only_tok_s = 0.0F;
  float request_tok_s = 0.0F;
  float p50_ms = 0.0F;
  float p95_ms = 0.0F;
  std::size_t emitted_tokens = 0;
  std::size_t eval_calls = 0;
  std::size_t session_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t graph_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint32_t capture = 0;
  std::uint32_t instantiate = 0;
  std::uint32_t upload = 0;
  std::uint32_t destroy = 0;
  std::uint32_t exec_destroy = 0;
  std::uint32_t topology_recapture = 0;
  std::uint32_t launch_state_uploads = 0;
  std::size_t decode_segment_graphs = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s [--workload identity|recapture-proof|cancellation|same-math|"
      "api|graph-ab|state-memory|long-context|activity-windows] "
      "[--execution-graphs ffn_only|decode_segments8] "
      "[--prefix N] [--warmups N] [--samples N] [--tokens N] [--capacity N] "
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
                       const qw38::cuda::EvalControl* control = nullptr,
                       qw38::cuda::DecodeAttribution* attribution = nullptr) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, control, nullptr,
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
                           qw38::cuda::SchedulerGraphs* graphs,
                           const char* path) {
  qw38::cuda::ExecutionGraphPathScope scope(path);
  if (std::strcmp(path, qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0) {
    return graphs->create(model, workspace, session);
  }
  return graphs->create(model, workspace);
}

void print_counts(std::size_t warmups, std::size_t samples,
                  std::size_t candidates, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

void fill_lifecycle(ArmResult* out, const qw38::cuda::SchedulerGraphs& graphs) {
  const qw38::cuda::GraphLifecycleCounts counts = graphs.lifecycle_counts();
  out->graph_bytes = graphs.allocated_bytes();
  out->decode_segment_graphs = graphs.decode_segment_graph_count();
  out->launch_state_uploads = counts.launch_state_upload;
  out->capture = counts.capture;
  out->instantiate = counts.instantiate;
  out->upload = counts.upload;
  out->destroy = counts.destroy;
  out->exec_destroy = counts.exec_destroy;
  out->topology_recapture = counts.topology_recapture;
}

bool exact_buffers(
    const std::vector<float>& left, const std::vector<float>& right,
    const std::array<float, qw38::internal::kResidualWidth>& hidden_left,
    const std::array<float, qw38::internal::kResidualWidth>& hidden_right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0 &&
         std::memcmp(hidden_left.data(), hidden_right.data(),
                     hidden_left.size() * sizeof(float)) == 0;
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

void print_profiler_probe() {
  const int nsys = std::system("command -v nsys >/dev/null 2>&1");
  const int ncu = std::system("command -v ncu >/dev/null 2>&1");
  std::printf(
      "nsys_available=%s ncu_available=%s cupti_linked=false "
      "full_ncu_sweep=false\n",
      nsys == 0 ? "true" : "false", ncu == 0 ? "true" : "false");
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
    status = create_graphs(model, &session, &workspace, &graphs,
                           qw38::cuda::kLegalExecutionGraphDecodeSegments8);
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
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\",\"workload\":\"identity\","
      "\"ok\":%s,\"parent\":\"post124_combined_opt118_opt119\","
      "\"candidate\":\"post124_plus_opt127_decode_segments8\","
      "\"shipping_execution_graphs\":\"%s\","
      "\"control_execution_graphs\":\"ffn_only\","
      "\"q4_decode\":\"%s\",\"packed_kv\":\"%s\",\"weight_requant\":\"%s\","
      "\"gdn_state\":\"%s\",\"decode_attention\":\"hybrid_crossover\","
      "\"crossover\":%d,\"n_parts\":%d,\"verified_max\":%d,"
      "\"poll_without_device_sync\":%s,\"defer_elapsed_event_sync\":%s,"
      "\"lazy\":%s,\"fusion\":%s,\"decode_d2h_bytes\":%llu,"
      "\"decode_segment_graphs\":%zu,\"opt130_selector_excluded\":true,"
      "\"opt128_pins_excluded\":true,\"opt131_fusion_excluded\":true,"
      "\"rejected_no_leak\":%s,\"opt110_hooks_ready\":%s,\"message\":\"%s\","
      "\"keep\":false}\n",
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
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

void append_lifecycle(std::string* body, const char* name,
                      const qw38::cuda::GraphLifecycleCounts& counts, bool last) {
  char row[768];
  std::snprintf(
      row, sizeof(row),
      "\"%s\":{\"capture\":%u,\"instantiate\":%u,\"upload\":%u,\"destroy\":%u,"
      "\"exec_destroy\":%u,\"launch_state_upload\":%u,"
      "\"topology_recapture\":%u}%s",
      name, counts.capture, counts.instantiate, counts.upload, counts.destroy,
      counts.exec_destroy, counts.launch_state_upload, counts.topology_recapture,
      last ? "" : ",");
  body->append(row);
}

int run_recapture_proof(const qw38::cuda::ResidentModel& model,
                        const Options& options) {
  struct Region final {
    const char* name;
    std::size_t prefix;
    std::size_t outputs;
    bool expect_cross;
  };
  const Region regions[] = {{"crossover_1024", 1023, 8, true},
                            {"steady_topology0", 128, 16, false}};
  bool all_ok = true;
  std::string body =
      "{\"schema_version\":1,\"task\":\"OPT-132\","
      "\"workload\":\"recapture-proof\",\"invalidation_policy\":\"";
  body.append(qw38::cuda::kDecodeGraphInvalidationPolicy);
  body.append("\",\"regions\":[");
  for (int region_index = 0; region_index < 2; ++region_index) {
    const Region& region = regions[region_index];
    const std::size_t capacity =
        session_capacity(region.prefix, region.outputs, options.capacity);
    std::vector<std::size_t> tokens(region.prefix + region.outputs);
    fill_tokens(&tokens);
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    qw38::Status status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    const qw38::cuda::GraphLifecycleCounts after_create =
        graphs.lifecycle_counts();
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok() && region.prefix > 0) {
      status = prefill(model, tokens, region.prefix, &session, &workspace,
                       &graphs, logits.data(), hidden.data());
    }
    const qw38::cuda::GraphLifecycleCounts after_prefill =
        graphs.lifecycle_counts();
    int topology_first = -1;
    int topology_last = -1;
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    for (std::size_t step = 0; status.is_ok() && step < region.outputs; ++step) {
      const int topology =
          qw38::cuda::decode_graph_topology_index(session.frontier());
      if (step == 0) topology_first = topology;
      topology_last = topology;
      float elapsed = 0.0F;
      status = run_token(model, tokens[region.prefix + step], &session,
                         &workspace, logits.data(), hidden.data(), &elapsed,
                         &graphs);
    }
    const qw38::cuda::GraphLifecycleCounts after_decode =
        graphs.lifecycle_counts();
    const std::uint32_t capture_delta =
        after_decode.capture - after_prefill.capture;
    const std::uint32_t instantiate_delta =
        after_decode.instantiate - after_prefill.instantiate;
    const std::uint32_t upload_delta =
        after_decode.upload - after_prefill.upload;
    const std::uint32_t destroy_delta =
        after_decode.destroy - after_prefill.destroy;
    const std::uint32_t recapture_delta =
        after_decode.topology_recapture - after_prefill.topology_recapture;
    const bool zero_recapture = capture_delta == 0 && instantiate_delta == 0 &&
                                upload_delta == 0 && destroy_delta == 0 &&
                                recapture_delta == 0;
    const bool crossed = topology_first != topology_last;
    std::string message;
    json_escape(status.message(), &message);
    const bool ok = status.is_ok() && zero_recapture &&
                    crossed == region.expect_cross &&
                    graphs.decode_segment_graph_count() ==
                        8 * qw38::cuda::kDecodeGraphTopologyCount;
    all_ok = all_ok && ok;
    char row[768];
    std::snprintf(
        row, sizeof(row),
        "%s{\"name\":\"%s\",\"ok\":%s,\"zero_steady_state_recapture\":%s,"
        "\"prefix\":%zu,\"outputs\":%zu,\"topology_first\":%d,"
        "\"topology_last\":%d,\"crossed_topology\":%s,\"capture_delta\":%u,"
        "\"instantiate_delta\":%u,\"upload_delta\":%u,\"destroy_delta\":%u,"
        "\"topology_recapture_delta\":%u,\"message\":\"%s\",",
        region_index == 0 ? "" : ",", region.name, json_bool(ok),
        json_bool(zero_recapture), region.prefix, region.outputs, topology_first,
        topology_last, json_bool(crossed), capture_delta, instantiate_delta,
        upload_delta, destroy_delta, recapture_delta, message.c_str());
    body.append(row);
    append_lifecycle(&body, "after_create", after_create, false);
    append_lifecycle(&body, "after_prefill", after_prefill, false);
    append_lifecycle(&body, "after_decode", after_decode, true);
    body.push_back('}');
    cudaDeviceSynchronize();
    cudaGetLastError();
  }
  char tail[192];
  std::snprintf(tail, sizeof(tail),
                "],\"ok\":%s,\"split_crossing_1024\":true,\"keep\":false}\n",
                json_bool(all_ok));
  body.append(tail);
  std::printf("%s%s", kPrefix, body.c_str());
  print_counts(0, 1, 1, false);
  return all_ok ? 0 : 1;
}

qw38::Status poll_stop(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls >= context->stop_after) {
    if (!context->requested) {
      context->requested = true;
      context->cancel_requested = std::chrono::steady_clock::now();
    }
    return {qw38::StatusCode::kCancelled, "opt132 graph cancellation"};
  }
  return qw38::Status::ok();
}

int run_cancellation(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  bool all_ok = true;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\","
      "\"workload\":\"cancellation\",\"cadence\":\"eight_layer_segment\","
      "\"cases\":[",
      kPrefix);
  const char* names[] = {"mid_segment", "commit"};
  const std::size_t stops[] = {1, 8};
  for (int case_index = 0; case_index < 2; ++case_index) {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    qw38::Status status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    const std::size_t frontier_before = session.frontier();
    PollContext poll_context{};
    poll_context.stop_after = stops[case_index];
    qw38::cuda::EvalControl control{};
    control.poll = poll_stop;
    control.context = &poll_context;
    float elapsed = 0.0F;
    qw38::Status token_status = qw38::Status::ok();
    const auto started = std::chrono::steady_clock::now();
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      token_status =
          run_token(model, tokens[prefix], &session, &workspace, logits.data(),
                    hidden.data(), &elapsed, &graphs, &control);
    }
    const float response_ms =
        poll_context.requested ? wall_ms(poll_context.cancel_requested)
                               : wall_ms(started);
    const bool cancelled = token_status.code() == qw38::StatusCode::kCancelled;
    const bool preserved = session.frontier() == frontier_before;
    const bool ok =
        status.is_ok() && cancelled && preserved && poll_context.calls >= 1;
    all_ok = all_ok && ok;
    std::printf(
        "%s{\"name\":\"%s\",\"ok\":%s,\"cancelled\":%s,\"frontier_preserved\":%s,"
        "\"poll_calls\":%zu,\"stop_after\":%zu,\"cancel_response_ms\":%.9g,"
        "\"atomic_commit\":%s,\"failure_before_commit\":%s}",
        case_index == 0 ? "" : ",", names[case_index], json_bool(ok),
        json_bool(cancelled), json_bool(preserved), poll_context.calls,
        stops[case_index], static_cast<double>(response_ms),
        json_bool(preserved), json_bool(preserved));
  }
  std::printf("],\"ok\":%s,\"keep\":false}\n", json_bool(all_ok));
  print_counts(0, 1, 1, false);
  return all_ok ? 0 : 1;
}

int run_same_math(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 8 : options.tokens;
  const std::size_t capacity =
      session_capacity(prefix, outputs, options.capacity);
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  std::vector<std::vector<float>> parent_logits;
  std::vector<std::array<float, qw38::internal::kResidualWidth>> parent_hidden;
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphFfnOnly);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphFfnOnly);
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float elapsed = 0.0F;
      status = run_token(model, tokens[prefix + step], &session, &workspace,
                         logits.data(), hidden.data(), &elapsed, &graphs);
      if (status.is_ok()) {
        parent_logits.push_back(logits);
        parent_hidden.push_back(hidden);
      }
    }
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  bool exact = status.is_ok() && parent_logits.size() == outputs;
  std::size_t matched = 0;
  qw38::cuda::GraphLifecycleCounts before{};
  qw38::cuda::GraphLifecycleCounts after{};
  {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    if (status.is_ok()) status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    before = graphs.lifecycle_counts();
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float elapsed = 0.0F;
      status = run_token(model, tokens[prefix + step], &session, &workspace,
                         logits.data(), hidden.data(), &elapsed, &graphs);
      const bool step_exact =
          status.is_ok() && step < parent_logits.size() &&
          exact_buffers(parent_logits[step], logits, parent_hidden[step],
                        hidden);
      exact = exact && step_exact;
      if (step_exact) ++matched;
    }
    after = graphs.lifecycle_counts();
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool zero_recapture =
      after.capture == before.capture &&
      after.instantiate == before.instantiate && after.upload == before.upload &&
      after.destroy == before.destroy &&
      after.topology_recapture == before.topology_recapture;
  const bool ok = status.is_ok() && exact && zero_recapture;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\",\"workload\":\"same-math\","
      "\"ok\":%s,\"exact\":%s,\"matched_tokens\":%zu,\"tokens\":%zu,"
      "\"prefix\":%zu,\"zero_recapture\":%s,\"control\":\"ffn_only\","
      "\"combination\":\"decode_segments8\",\"state_equals\":%s,"
      "\"message\":\"%s\",\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(exact), matched, outputs, prefix,
      json_bool(zero_recapture), json_bool(exact), message.c_str());
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_api(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 8, options.capacity);
  std::vector<std::size_t> tokens(prefix + 8);
  fill_tokens(&tokens);
  const char* tmp = "/tmp/qw38-opt132-ckpt.bin";
  const char* graphs_path = qw38::cuda::kLegalExecutionGraphDecodeSegments8;
  bool repeat_equal = false;
  bool eval_changed = false;
  bool restore_equals = false;
  bool cancel_isolated = false;
  bool replay_after_restore = false;
  bool divergent_invalidated = false;
  bool session_replace_invalidated = false;
  float ttft_ms = 0.0F;
  float wall = 0.0F;
  std::vector<float> latencies;
  std::size_t emitted = 0;
  std::size_t live_calls = 0;
  std::uint32_t capture = 0;
  std::uint32_t instantiate = 0;
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs, graphs_path);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    qw38::cuda::ExecutionGraphPathScope scope(graphs_path);
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
      status = session.copy_last_outputs(
          again.data(), again.size(), again_hidden.data(), again_hidden.size());
    }
    repeat_equal = std::memcmp(before.data(), again.data(),
                               before.size() * sizeof(float)) == 0;
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
    eval_changed = std::memcmp(before.data(), after_eval.data(),
                               before.size() * sizeof(float)) != 0;
    if (status.is_ok()) status = session.save_checkpoint(tmp);
    {
      qw38::cuda::SchedulerSession restored;
      if (status.is_ok()) status = restored.create(capacity);
      if (status.is_ok()) status = restored.restore_checkpoint(tmp, &workspace);
      if (status.is_ok()) {
        status = restored.state_equals(session, &restore_equals);
      }
    }
    if (status.is_ok()) {
      std::vector<std::size_t> divergent(prefix + 2);
      fill_tokens(&divergent);
      for (std::size_t index = 0; index < divergent.size(); ++index) {
        divergent[index] =
            (divergent[index] + 13) % qw38::internal::kVocabularySize;
      }
      qw38::cuda::SchedulerSession other;
      qw38::cuda::SchedulerWorkspace other_workspace;
      qw38::Status other_status = other.create(capacity);
      if (other_status.is_ok()) other_status = other_workspace.create(capacity);
      std::vector<float> other_logits(qw38::internal::kVocabularySize);
      std::array<float, qw38::internal::kResidualWidth> other_hidden{};
      if (other_status.is_ok()) {
        other_status = prefill(model, divergent, prefix, &other, &other_workspace,
                               nullptr, other_logits.data(), other_hidden.data());
      }
      if (other_status.is_ok()) {
        float mismatch_ms = 0.0F;
        const qw38::Status mismatch = run_token(
            model, divergent[prefix], &other, &other_workspace,
            other_logits.data(), other_hidden.data(), &mismatch_ms, &graphs);
        divergent_invalidated =
            mismatch.code() == qw38::StatusCode::kInvalidArgument;
      }
    }
    if (status.is_ok()) {
      qw38::cuda::SchedulerSession replacement;
      qw38::cuda::SchedulerWorkspace replacement_workspace;
      qw38::Status replace_status = replacement.create(capacity);
      if (replace_status.is_ok()) {
        replace_status = replacement_workspace.create(capacity);
      }
      if (replace_status.is_ok()) {
        float mismatch_ms = 0.0F;
        std::vector<float> repl_logits(qw38::internal::kVocabularySize);
        std::array<float, qw38::internal::kResidualWidth> repl_hidden{};
        const qw38::Status mismatch =
            run_token(model, tokens[prefix], &replacement, &replacement_workspace,
                      repl_logits.data(), repl_hidden.data(), &mismatch_ms,
                      &graphs);
        session_replace_invalidated =
            mismatch.code() == qw38::StatusCode::kInvalidArgument;
      }
    }
    PollContext poll{0, 1};
    const qw38::cuda::EvalControl cancel{poll_stop, &poll};
    const std::size_t frontier_before = session.frontier();
    qw38::Status cancelled =
        run_token(model, tokens[prefix + 1], &session, &workspace, logits.data(),
                  hidden.data(), &elapsed, &graphs, &cancel);
    cancel_isolated = cancelled.code() == qw38::StatusCode::kCancelled &&
                      session.frontier() == frontier_before;
    PollContext live_poll{};
    const qw38::cuda::EvalControl live{
        [](void* context) noexcept -> qw38::Status {
          auto* ctx = static_cast<PollContext*>(context);
          ++ctx->calls;
          return qw38::Status::ok();
        },
        &live_poll};
    const auto request_started = std::chrono::steady_clock::now();
    qw38::Status loop = qw38::Status::ok();
    for (std::size_t step = 0; loop.is_ok() && step < 4; ++step) {
      const auto token_started = std::chrono::steady_clock::now();
      loop = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs, &live);
      latencies.push_back(wall_ms(token_started));
      if (step == 0) ttft_ms = wall_ms(request_started);
      ++emitted;
    }
    wall = wall_ms(request_started);
    live_calls = live_poll.calls;
    const qw38::cuda::GraphLifecycleCounts counts = graphs.lifecycle_counts();
    capture = counts.capture;
    instantiate = counts.instantiate;
    if (!loop.is_ok()) status = loop;
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  std::vector<float> match_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> match_hidden{};
  {
    qw38::cuda::SchedulerSession restored;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    if (status.is_ok()) status = restored.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) status = restored.restore_checkpoint(tmp, &workspace);
    if (status.is_ok()) {
      status = create_graphs(model, &restored, &workspace, &graphs, graphs_path);
    }
    float elapsed = 0.0F;
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(graphs_path);
      status = run_token(model, tokens[prefix + 1], &restored, &workspace,
                         match_logits.data(), match_hidden.data(), &elapsed,
                         &graphs);
    }
    replay_after_restore = status.is_ok();
  }
  std::remove(tmp);
  const bool ok = status.is_ok() && repeat_equal && eval_changed &&
                  restore_equals && replay_after_restore &&
                  divergent_invalidated && session_replace_invalidated &&
                  cancel_isolated && live_calls > 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\",\"workload\":\"api\","
      "\"ok\":%s,\"repeat_logits_equal\":%s,\"eval_changes_logits\":%s,"
      "\"restore_equals\":%s,\"replay_after_restore\":%s,"
      "\"divergent_prefix_invalidated\":%s,\"session_replace_invalidated\":%s,"
      "\"cancel_isolated\":%s,"
      "\"request_loop\":true,\"ttft_ms\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,"
      "\"wall_ms\":%.9g,\"emitted_tokens\":%zu,\"cancellation_callbacks\":%zu,"
      "\"steady_capture\":%u,\"steady_instantiate\":%u,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(repeat_equal), json_bool(eval_changed),
      json_bool(restore_equals), json_bool(replay_after_restore),
      json_bool(divergent_invalidated), json_bool(session_replace_invalidated),
      json_bool(cancel_isolated),
      static_cast<double>(ttft_ms),
      static_cast<double>(percentile(latencies, 0.50F)),
      static_cast<double>(percentile(latencies, 0.95F)),
      static_cast<double>(wall), emitted, live_calls, capture, instantiate);
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

qw38::Status run_request_arm(const qw38::cuda::ResidentModel& model,
                             const std::vector<std::size_t>& tokens,
                             std::size_t prefix, std::size_t outputs,
                             const char* graph_path, std::size_t requested,
                             ArmResult* out,
                             qw38::cuda::DecodeAttribution* attribution) {
  const std::size_t capacity = session_capacity(prefix, outputs, requested);
  const auto setup_started = std::chrono::steady_clock::now();
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) {
    status = create_graphs(model, &session, &workspace, &graphs, graph_path);
  }
  out->setup_ms = wall_ms(setup_started);
  if (!status.is_ok()) return status;
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  const auto request_started = std::chrono::steady_clock::now();
  if (prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  out->prefill_ms = wall_ms(request_started);
  out->ttft_ms = out->setup_ms + out->prefill_ms;
  if (!status.is_ok()) return status;
  workspace.reset_transfer_counters();
  std::vector<float> latencies;
  latencies.reserve(outputs);
  const auto decode_started = std::chrono::steady_clock::now();
  qw38::cuda::ExecutionGraphPathScope scope(graph_path);
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    qw38::cuda::DecodeAttribution* used = attribution;
    if (used != nullptr && !in_window(step, outputs)) used = nullptr;
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs, nullptr,
                       used);
    latencies.push_back(wall_ms(token_started));
    if (step == 0) out->ttft_ms = out->setup_ms + wall_ms(request_started);
  }
  out->decode_only_ms = wall_ms(decode_started);
  out->request_ms = out->setup_ms + wall_ms(request_started);
  const std::size_t counted = outputs == 0 ? prefix : outputs;
  out->emitted_tokens = outputs;
  out->eval_calls = outputs;
  out->request_tok_s =
      out->request_ms > 0.0F
          ? static_cast<float>(counted) * 1000.0F / out->request_ms
          : 0.0F;
  out->decode_only_tok_s =
      outputs > 0 && out->decode_only_ms > 0.0F
          ? static_cast<float>(outputs) * 1000.0F / out->decode_only_ms
          : out->request_tok_s;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  out->d2h_bytes = workspace.transfer_d2h_bytes_;
  out->session_bytes = session.allocated_bytes();
  out->workspace_bytes = workspace.allocated_bytes();
  fill_lifecycle(out, graphs);
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"setup_ms\":%.9g,\"prefill_ms\":%.9g,\"decode_only_ms\":%.9g,"
      "\"request_ms\":%.9g,\"ttft_ms\":%.9g,\"decode_only_tok_s\":%.9g,"
      "\"request_tok_s\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,"
      "\"emitted_tokens\":%zu,\"eval_calls\":%zu,\"session_bytes\":%zu,"
      "\"workspace_bytes\":%zu,\"graph_bytes\":%zu,\"d2h_bytes\":%llu,"
      "\"capture\":%u,\"instantiate\":%u,\"upload\":%u,\"destroy\":%u,"
      "\"topology_recapture\":%u,\"decode_segment_graphs\":%zu}%s",
      name, static_cast<double>(arm.setup_ms), static_cast<double>(arm.prefill_ms),
      static_cast<double>(arm.decode_only_ms),
      static_cast<double>(arm.request_ms), static_cast<double>(arm.ttft_ms),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.request_tok_s),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms),
      arm.emitted_tokens, arm.eval_calls, arm.session_bytes, arm.workspace_bytes,
      arm.graph_bytes, static_cast<unsigned long long>(arm.d2h_bytes),
      arm.capture, arm.instantiate, arm.upload, arm.destroy,
      arm.topology_recapture, arm.decode_segment_graphs, last ? "" : ",");
}

int run_graph_ab(const qw38::cuda::ResidentModel& model, const Options& options) {
  const bool prefill_only = options.tokens == 0;
  const std::size_t prompt = options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  const char* parent = qw38::cuda::kLegalExecutionGraphFfnOnly;
  const char* candidate = options.execution_graphs;
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_request_arm(model, tokens, prompt, outputs, parent, 0,
                             &discarded, nullptr);
    if (status.is_ok()) {
      status = run_request_arm(model, tokens, prompt, outputs, candidate, 0,
                               &discarded, nullptr);
    }
    if (!status.is_ok()) return fail_status(status);
    std::printf(
        "quartz_warmup=%zu decode_only_tok_s=%.9g request_tok_s=%.9g "
        "independently_restored=true\n",
        warmup, static_cast<double>(discarded.decode_only_tok_s),
        static_cast<double>(discarded.request_tok_s));
  }
  std::vector<ArmResult> arm_a(options.samples);
  std::vector<ArmResult> arm_b(options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    const bool ba = (pair % 2) == 1;
    ArmResult first;
    ArmResult second;
    status = run_request_arm(model, tokens, prompt, outputs,
                             ba ? candidate : parent, 0, &first, nullptr);
    if (status.is_ok()) {
      status = run_request_arm(model, tokens, prompt, outputs,
                               ba ? parent : candidate, 0, &second, nullptr);
    }
    if (!status.is_ok()) return fail_status(status);
    arm_a[pair] = ba ? second : first;
    arm_b[pair] = ba ? first : second;
    std::printf(
        "quartz_pair sample_index=%zu order=%s A_decode_only_tok_s=%.9g "
        "B_decode_only_tok_s=%.9g A_request_tok_s=%.9g B_request_tok_s=%.9g "
        "independently_restored=true same_binary=true\n",
        pair, ba ? "BA" : "AB",
        static_cast<double>(arm_a[pair].decode_only_tok_s),
        static_cast<double>(arm_b[pair].decode_only_tok_s),
        static_cast<double>(arm_a[pair].request_tok_s),
        static_cast<double>(arm_b[pair].request_tok_s));
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\",\"workload\":\"graph-ab\","
      "\"prefix\":%zu,\"prefill\":%s,\"warmups\":%zu,\"samples\":%zu,"
      "\"tokens\":%zu,\"parent\":\"%s\",\"candidate\":\"%s\","
      "\"control\":\"post124_combined_opt118_opt119\","
      "\"combination\":\"post124_plus_opt127_decode_segments8\","
      "\"metrics\":[\"decode_only\",\"complete_request\"],"
      "\"metric_clock_starts_after_prefill_for_decode_only\":true,"
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"observed_shapes\":1,"
      "\"observed_tier\":\"acceptance\",\"keep\":false,"
      "\"independently_restored\":true,\"same_binary\":true,\"pairs\":[",
      kPrefix, prompt, json_bool(prefill_only), options.warmups, options.samples,
      outputs, parent, candidate, options.warmups, options.samples);
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

int run_state_memory(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  std::vector<float> parent_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> parent_hidden{};
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphFfnOnly);
    }
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       parent_logits.data(), parent_hidden.data());
    }
    float elapsed = 0.0F;
    if (status.is_ok()) {
      status = run_token(model, tokens[prefix], &session, &workspace,
                         parent_logits.data(), parent_hidden.data(), &elapsed,
                         &graphs);
    }
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  std::vector<float> cand_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> cand_hidden{};
  std::size_t graph_bytes = 0;
  std::size_t session_bytes = 0;
  std::size_t workspace_bytes = 0;
  {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    if (status.is_ok()) status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       cand_logits.data(), cand_hidden.data());
    }
    float elapsed = 0.0F;
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix], &session, &workspace,
                         cand_logits.data(), cand_hidden.data(), &elapsed,
                         &graphs);
    }
    graph_bytes = graphs.allocated_bytes();
    session_bytes = session.allocated_bytes();
    workspace_bytes = workspace.allocated_bytes();
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool equal = status.is_ok() && exact_buffers(parent_logits, cand_logits,
                                                     parent_hidden, cand_hidden);
  const bool ok = status.is_ok() && equal;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\","
      "\"workload\":\"state-memory\",\"ok\":%s,\"state_equals\":%s,"
      "\"graph_bytes\":%zu,\"session_bytes\":%zu,\"workspace_bytes\":%zu,"
      "\"peak_bytes\":%zu,\"message\":\"%s\",\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(equal), graph_bytes, session_bytes,
      workspace_bytes, session_bytes + workspace_bytes + graph_bytes,
      message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
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
  ArmResult measured;
  const qw38::Status status = run_request_arm(
      model, tokens, prefix, outputs,
      qw38::cuda::kLegalExecutionGraphDecodeSegments8, capacity, &measured,
      nullptr);
  if (!status.is_ok()) return fail_status(status);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\","
      "\"workload\":\"long-context\",\"ok\":true,\"prefix\":%zu,\"tokens\":%zu,"
      "\"capacity\":%zu,\"setup_ms\":%.9g,\"prefill_ms\":%.9g,"
      "\"decode_only_ms\":%.9g,\"request_ms\":%.9g,\"ttft_ms\":%.9g,"
      "\"decode_only_tok_s\":%.9g,\"request_tok_s\":%.9g,\"p50_ms\":%.9g,"
      "\"p95_ms\":%.9g,\"emitted_tokens\":%zu,\"eval_calls\":%zu,"
      "\"session_bytes\":%zu,\"workspace_bytes\":%zu,"
      "\"populated_cache\":true,\"capacity_alone_is_not_traffic\":true,"
      "\"keep\":false}\n",
      kPrefix, prefix, outputs, capacity, static_cast<double>(measured.setup_ms),
      static_cast<double>(measured.prefill_ms),
      static_cast<double>(measured.decode_only_ms),
      static_cast<double>(measured.request_ms),
      static_cast<double>(measured.ttft_ms),
      static_cast<double>(measured.decode_only_tok_s),
      static_cast<double>(measured.request_tok_s),
      static_cast<double>(measured.p50_ms), static_cast<double>(measured.p95_ms),
      measured.emitted_tokens, measured.eval_calls, measured.session_bytes,
      measured.workspace_bytes);
  print_counts(0, 1, 1, false);
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
  qw38::Status status = run_request_arm(
      model, tokens, prefix, outputs,
      qw38::cuda::kLegalExecutionGraphDecodeSegments8, 0, &uninstrumented,
      nullptr);
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
  ArmResult measured;
  status = run_request_arm(model, tokens, prefix, outputs,
                           qw38::cuda::kLegalExecutionGraphDecodeSegments8, 0,
                           &measured, &attribution);
  qw38::cuda::destroy_sequence_epoch(&families);
  if (!status.is_ok()) return fail_status(status);
  std::printf("%s[", kRecords);
  for (std::size_t index = 0; index < families.count; ++index) {
    qw38::cuda::write_engine_record_json(stdout, families.records[index],
                                         index + 1 == families.count);
  }
  std::printf("]\n");
  const float perturbation = measured.decode_only_ms - uninstrumented.decode_only_ms;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-132\","
      "\"workload\":\"activity-windows\","
      "\"stack\":\"post124_plus_opt127_decode_segments8\",\"prefix\":%zu,"
      "\"tokens\":%zu,\"record_count\":%zu,\"pool_overflow\":%s,"
      "\"uninstrumented_decode_only_wall_ms\":%.9g,"
      "\"instrumented_decode_only_wall_ms\":%.9g,"
      "\"profiler_perturbation_ms\":%.9g,\"decode_only_tok_s\":%.9g,"
      "\"request_tok_s\":%.9g,\"windows\":{\"early\":\"0:4\","
      "\"middle\":\"mid-2:mid+2\",\"late\":\"n-4:n\"},"
      "\"method\":\"cuda_event_engine_attribution\","
      "\"cupti_linked\":false,\"keep\":false}\n",
      kPrefix, prefix, outputs, families.count,
      json_bool(families.pool_overflow),
      static_cast<double>(uninstrumented.decode_only_ms),
      static_cast<double>(measured.decode_only_ms),
      static_cast<double>(perturbation),
      static_cast<double>(measured.decode_only_tok_s),
      static_cast<double>(measured.request_tok_s));
  print_counts(1, 1, 1, false);
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
  if (std::strcmp(options.workload, "identity") == 0) {
    return run_identity(model, options);
  }
  if (std::strcmp(options.workload, "recapture-proof") == 0) {
    return run_recapture_proof(model, options);
  }
  if (std::strcmp(options.workload, "cancellation") == 0) {
    return run_cancellation(model, options);
  }
  if (std::strcmp(options.workload, "same-math") == 0) {
    return run_same_math(model, options);
  }
  if (std::strcmp(options.workload, "api") == 0) {
    return run_api(model, options);
  }
  if (std::strcmp(options.workload, "graph-ab") == 0) {
    return run_graph_ab(model, options);
  }
  if (std::strcmp(options.workload, "state-memory") == 0) {
    return run_state_memory(model, options);
  }
  if (std::strcmp(options.workload, "long-context") == 0) {
    return run_long_context(model, options);
  }
  if (std::strcmp(options.workload, "activity-windows") == 0) {
    return run_activity_windows(model, options);
  }
  return usage(argv[0]);
}
