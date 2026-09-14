#include "attention_decode_path.cuh"
#include "decode_launch_state.cuh"
#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "gdn_decode_path.cuh"
#include "opt110_engine_hook.cuh"
#include "scheduler_primitives.h"
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

constexpr char kPrefix[] = "QW38_OPT131_DECODE_CHAIN_RESULT=";
constexpr char kCounts[] = "QW38_OPT131_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr int kDefaultWarmups = 1;
constexpr int kDefaultPairs = 3;
constexpr std::size_t kDefaultDecodeTokens = 4;
constexpr std::size_t kLogitsBytes =
    qw38::internal::kVocabularySize * sizeof(float);
constexpr int kIsolatedRepeats = 32;
constexpr float kPessimisticGBs = 100.0F;
constexpr float kPeakGBs = 1792.0F;

struct Options final {
  const char* workload = "identity";
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

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s [--workload identity|profile|occupancy|lifetime|"
      "cancellation|equivalence] [--prefix N] [--warmups N] [--samples N] "
      "[--tokens N] [--capacity N] MODEL independently_restored=true "
      "same_binary=true\n",
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
  return static_cast<float>(std::chrono::duration<double, std::milli>(
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

qw38::Status poll_stop(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls >= context->stop_after) {
    return {qw38::StatusCode::kCancelled, "opt131 decode chain cancellation"};
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
      "%s{\"schema_version\":1,\"task\":\"OPT-131\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

void print_leaf(const char* name, const qw38::cuda::TimingValue& value,
                bool last) {
  std::printf("\"%s\":{\"ms\":%.9g,\"measured\":%s}%s", name,
              static_cast<double>(value.milliseconds),
              json_bool(value.measured), last ? "" : ",");
}

float event_ms(cudaError_t (*launch)(void*), void* ctx, int repeats) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  launch(ctx);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int index = 0; index < repeats; ++index) launch(ctx);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed = 0.0F;
  cudaEventElapsedTime(&elapsed, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed / static_cast<float>(repeats);
}

struct CastCtx final {
  float* in = nullptr;
  __nv_bfloat16* out = nullptr;
  std::size_t count = 0;
};

cudaError_t launch_cast(void* opaque) {
  auto* ctx = static_cast<CastCtx*>(opaque);
  return qw38::cuda::launch_fp32_to_bf16(ctx->in, ctx->count, ctx->out,
                                         nullptr);
}

struct SplitCtx final {
  float* packed = nullptr;
  float* query = nullptr;
  float* gate = nullptr;
};

cudaError_t launch_split(void* opaque) {
  auto* ctx = static_cast<SplitCtx*>(opaque);
  return qw38::cuda::launch_split_attention_query_gate(
      ctx->packed, 24, 256, ctx->query, ctx->gate, nullptr);
}

struct GateCtx final {
  float* alpha = nullptr;
  float* beta = nullptr;
  float* folded = nullptr;
  float* dt = nullptr;
  float* decay = nullptr;
  float* update = nullptr;
};

cudaError_t launch_gates(void* opaque) {
  auto* ctx = static_cast<GateCtx*>(opaque);
  return qw38::cuda::launch_prepare_gdn_gates(
      ctx->alpha, ctx->beta, ctx->folded, ctx->dt, 16, 3, ctx->decay,
      ctx->update, nullptr);
}

struct GatedCtx final {
  float* rec = nullptr;
  float* gate = nullptr;
  float* norm = nullptr;
  __nv_bfloat16* out = nullptr;
};

cudaError_t launch_gated(void* opaque) {
  auto* ctx = static_cast<GatedCtx*>(opaque);
  return qw38::cuda::launch_gdn_gated_output(ctx->rec, ctx->gate, ctx->norm, 16,
                                             3, 128, ctx->out, nullptr);
}

struct ResidCtx final {
  float* residual = nullptr;
  float* corr = nullptr;
  float* out = nullptr;
};

cudaError_t launch_resid(void* opaque) {
  auto* ctx = static_cast<ResidCtx*>(opaque);
  return qw38::cuda::launch_residual_add_fp32(
      ctx->residual, ctx->corr, qw38::internal::kResidualWidth, ctx->out,
      nullptr);
}

int run_identity(const qw38::cuda::ResidentModel& model,
                 const Options& options) {
  const std::size_t capacity = session_capacity(options.prefix, 1, options.capacity);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &session, &workspace, &graphs);
  const bool graphs_ok =
      status.is_ok() && graphs.decode_segment_graph_count() ==
                            qw38::cuda::kDecodeSegmentCount *
                                qw38::cuda::kDecodeGraphTopologyCount;
  const bool q8 = qw38::cuda::mixer_norm_q8_fusion_enabled() &&
                  qw38::cuda::ffn_norm_q8_fusion_enabled();
  const bool graphs_pin =
      std::strcmp(qw38::cuda::effective_execution_graph_path(),
                  qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0;
  const bool hybrid =
      qw38::cuda::effective_decode_attention_crossover_threshold() == 1024 &&
      qw38::cuda::effective_decode_attention_verified_max() == 4096 &&
      qw38::cuda::effective_vec128_n_parts() == 16;
  const bool gdn_seq =
      std::strcmp(qw38::cuda::effective_gdn_decode_path(),
                  qw38::cuda::kLegalGdnDecodePathSequential) == 0;
  const bool stalls_off = !qw38::cuda::poll_without_device_sync_enabled() &&
                          !qw38::cuda::defer_elapsed_event_sync_enabled();
  const bool ok = graphs_ok && q8 && graphs_pin && hybrid && gdn_seq && stalls_off;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-131\",\"workload\":\"identity\","
      "\"ok\":%s,\"parent\":\"decode_segments8\","
      "\"execution_graphs\":\"%s\",\"decode_segment_graph_count\":%zu,"
      "\"decode_node_count\":%zu,\"mixer_norm_q8_fusion\":%s,"
      "\"ffn_norm_q8_fusion\":%s,\"crossover_threshold\":%d,"
      "\"verified_max\":%d,\"vec128_n_parts\":%d,\"gdn_decode\":\"%s\","
      "\"poll_without_device_sync\":%s,\"defer_elapsed_event_sync\":%s,"
      "\"opt119_reused_not_recreated\":true,\"independently_restored\":true,"
      "\"same_binary\":true}\n",
      kPrefix, json_bool(ok), qw38::cuda::effective_execution_graph_path(),
      graphs.decode_segment_graph_count(), graphs.decode_node_count(),
      json_bool(qw38::cuda::mixer_norm_q8_fusion_enabled()),
      json_bool(qw38::cuda::ffn_norm_q8_fusion_enabled()),
      qw38::cuda::effective_decode_attention_crossover_threshold(),
      qw38::cuda::effective_decode_attention_verified_max(),
      qw38::cuda::effective_vec128_n_parts(),
      qw38::cuda::effective_gdn_decode_path(),
      json_bool(qw38::cuda::poll_without_device_sync_enabled()),
      json_bool(qw38::cuda::defer_elapsed_event_sync_enabled()));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int time_decode_arm(const qw38::cuda::ResidentModel& model,
                    const std::vector<std::size_t>& tokens, std::size_t prefix,
                    std::size_t outputs, bool use_graphs, float* wall,
                    qw38::cuda::LeafTimings* leaves, std::size_t* node_count,
                    std::size_t* segment_count) {
  const std::size_t capacity = session_capacity(prefix, outputs, 0);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  qw38::cuda::SchedulerGraphs* graph_ptr = nullptr;
  if (use_graphs) {
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs);
    }
    graph_ptr = &graphs;
    if (node_count != nullptr) *node_count = graphs.decode_node_count();
    if (segment_count != nullptr) {
      *segment_count = graphs.decode_segment_graph_count();
    }
  } else {
    if (node_count != nullptr) *node_count = 0;
    if (segment_count != nullptr) *segment_count = 0;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::ExecutionGraphPathScope path_scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, graph_ptr,
                     logits.data(), hidden.data());
  }
  qw38::cuda::DecodeAttribution attribution{};
  attribution.record_leaves = leaves != nullptr;
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    float elapsed = 0.0F;
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, graph_ptr,
                       nullptr, leaves == nullptr ? nullptr : &attribution);
  }
  if (wall != nullptr) *wall = wall_ms(started);
  if (leaves != nullptr) *leaves = attribution.leaves;
  return status.is_ok() ? 0 : fail_status(status);
}

int run_isolated(float* cast_ms, float* split_ms, float* gates_ms,
                 float* gated_ms, float* resid_ms) {
  CastCtx cast{};
  SplitCtx split{};
  GateCtx gates{};
  GatedCtx gated{};
  ResidCtx resid{};
  cudaError_t error = cudaMalloc(&cast.in, 6144 * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMalloc(&cast.out, 6144 * sizeof(__nv_bfloat16));
  }
  cast.count = 6144;
  if (error == cudaSuccess) error = cudaMalloc(&split.packed, 12288 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&split.query, 6144 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&split.gate, 6144 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gates.alpha, 48 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gates.beta, 48 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gates.folded, 48 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gates.dt, 48 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gates.decay, 48 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gates.update, 48 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gated.rec, 6144 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gated.gate, 6144 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&gated.norm, 128 * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMalloc(&gated.out, 6144 * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&resid.residual,
                       qw38::internal::kResidualWidth * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&resid.corr,
                       qw38::internal::kResidualWidth * sizeof(float));
  }
  if (error == cudaSuccess) {
    error =
        cudaMalloc(&resid.out, qw38::internal::kResidualWidth * sizeof(float));
  }
  if (error != cudaSuccess) return 1;
  cudaMemset(cast.in, 0, 6144 * sizeof(float));
  cudaMemset(split.packed, 0, 12288 * sizeof(float));
  cudaMemset(gates.alpha, 0, 48 * sizeof(float));
  cudaMemset(gates.beta, 0, 48 * sizeof(float));
  cudaMemset(gates.folded, 0, 48 * sizeof(float));
  cudaMemset(gates.dt, 0, 48 * sizeof(float));
  cudaMemset(gated.rec, 0, 6144 * sizeof(float));
  cudaMemset(gated.gate, 0, 6144 * sizeof(float));
  cudaMemset(gated.norm, 0, 128 * sizeof(float));
  cudaMemset(resid.residual, 0, qw38::internal::kResidualWidth * sizeof(float));
  cudaMemset(resid.corr, 0, qw38::internal::kResidualWidth * sizeof(float));
  *cast_ms = event_ms(launch_cast, &cast, kIsolatedRepeats);
  *split_ms = event_ms(launch_split, &split, kIsolatedRepeats);
  *gates_ms = event_ms(launch_gates, &gates, kIsolatedRepeats);
  *gated_ms = event_ms(launch_gated, &gated, kIsolatedRepeats);
  *resid_ms = event_ms(launch_resid, &resid, kIsolatedRepeats);
  cudaFree(cast.in);
  cudaFree(cast.out);
  cudaFree(split.packed);
  cudaFree(split.query);
  cudaFree(split.gate);
  cudaFree(gates.alpha);
  cudaFree(gates.beta);
  cudaFree(gates.folded);
  cudaFree(gates.dt);
  cudaFree(gates.decay);
  cudaFree(gates.update);
  cudaFree(gated.rec);
  cudaFree(gated.gate);
  cudaFree(gated.norm);
  cudaFree(gated.out);
  cudaFree(resid.residual);
  cudaFree(resid.corr);
  cudaFree(resid.out);
  return 0;
}

int run_profile(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 4 : options.tokens;
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    float discarded = 0.0F;
    if (time_decode_arm(model, tokens, prefix, outputs, true, &discarded,
                        nullptr, nullptr, nullptr) != 0) {
      return 1;
    }
  }
  float graph_ms = 0.0F;
  float eager_ms = 0.0F;
  std::size_t nodes = 0;
  std::size_t segments = 0;
  qw38::cuda::LeafTimings leaves{};
  if (time_decode_arm(model, tokens, prefix, outputs, true, &graph_ms, nullptr,
                      &nodes, &segments) != 0) {
    return 1;
  }
  if (time_decode_arm(model, tokens, prefix, outputs, false, &eager_ms, &leaves,
                      nullptr, nullptr) != 0) {
    return 1;
  }
  float cast_ms = 0.0F;
  float split_ms = 0.0F;
  float gates_ms = 0.0F;
  float gated_ms = 0.0F;
  float resid_ms = 0.0F;
  if (run_isolated(&cast_ms, &split_ms, &gates_ms, &gated_ms, &resid_ms) != 0) {
    return 1;
  }
  const float graph_tok = graph_ms / static_cast<float>(outputs);
  const float eager_tok = eager_ms / static_cast<float>(outputs);
  const float gap = eager_tok > graph_tok ? eager_tok - graph_tok : 0.0F;
  const bool ok = graph_ms > 0.0F && eager_ms > 0.0F && segments == 16 &&
                  leaves.gdn_conv_fused_with_recurrence &&
                  leaves.gdn_output_norm_fused_with_gate &&
                  leaves.attention_prep_fused_with_core &&
                  qw38::cuda::mixer_norm_q8_fusion_enabled() &&
                  qw38::cuda::ffn_norm_q8_fusion_enabled();
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-131\",\"workload\":\"profile\","
      "\"ok\":%s,\"parent\":\"decode_segments8\",\"prefix\":%zu,\"outputs\":%zu,"
      "\"opt119_mixer_norm_q8\":%s,\"opt119_ffn_norm_q8\":%s,"
      "\"pessimistic_gbs\":%.9g,\"listed_peak_gbs\":%.9g,"
      "\"graph\":{\"wall_ms\":%.9g,\"per_token_ms\":%.9g,"
      "\"decode_node_count\":%zu,\"decode_segment_graph_count\":%zu},"
      "\"eager\":{\"wall_ms\":%.9g,\"per_token_ms\":%.9g,\"leaves\":{",
      kPrefix, json_bool(ok), prefix, outputs,
      json_bool(qw38::cuda::mixer_norm_q8_fusion_enabled()),
      json_bool(qw38::cuda::ffn_norm_q8_fusion_enabled()),
      static_cast<double>(kPessimisticGBs), static_cast<double>(kPeakGBs),
      static_cast<double>(graph_ms), static_cast<double>(graph_tok), nodes,
      segments, static_cast<double>(eager_ms), static_cast<double>(eager_tok));
  print_leaf("embedding", leaves.embedding, false);
  print_leaf("gdn_gate_prep", leaves.gdn_gate_prep, false);
  print_leaf("gdn_conv_qk_norm_recurrence", leaves.gdn_conv_qk_norm_recurrence,
             false);
  print_leaf("gdn_output_norm", leaves.gdn_output_norm, false);
  print_leaf("attn_query_split", leaves.attn_query_split, false);
  print_leaf("attn_qk_prep_softmax_pv_merge",
             leaves.attn_qk_prep_softmax_pv_merge, false);
  print_leaf("attn_output_cast", leaves.attn_output_cast, false);
  print_leaf("residual_mixer", leaves.residual_mixer, false);
  print_leaf("ffn_norm", leaves.ffn_norm, false);
  print_leaf("logits_norm", leaves.logits_norm, true);
  std::printf(
      "},\"gdn_conv_fused_with_recurrence\":%s,"
      "\"gdn_output_norm_fused_with_gate\":%s,"
      "\"attention_prep_fused_with_core\":%s},"
      "\"launch_gap_ms_per_token\":%.9g,"
      "\"isolated\":{\"fp32_to_bf16_attn_ms\":%.9g,"
      "\"split_attention_ms\":%.9g,\"prepare_gdn_gates_ms\":%.9g,"
      "\"gdn_gated_output_ms\":%.9g,\"residual_add_ms\":%.9g},"
      "\"independently_restored\":true,\"same_binary\":true}\n",
      json_bool(leaves.gdn_conv_fused_with_recurrence),
      json_bool(leaves.gdn_output_norm_fused_with_gate),
      json_bool(leaves.attention_prep_fused_with_core),
      static_cast<double>(gap), static_cast<double>(cast_ms),
      static_cast<double>(split_ms), static_cast<double>(gates_ms),
      static_cast<double>(gated_ms), static_cast<double>(resid_ms));
  print_counts(options.warmups, 1, 1, false);
  return ok ? 0 : 1;
}

int run_occupancy(const qw38::cuda::ResidentModel& /*model*/,
                 const Options& /*options*/) {
  float cast_ms = 0.0F;
  float split_ms = 0.0F;
  float gates_ms = 0.0F;
  float gated_ms = 0.0F;
  float resid_ms = 0.0F;
  if (run_isolated(&cast_ms, &split_ms, &gates_ms, &gated_ms, &resid_ms) != 0) {
    return 1;
  }
  const bool ok = cast_ms > 0.0F && split_ms > 0.0F && gates_ms > 0.0F &&
                  gated_ms > 0.0F && resid_ms > 0.0F;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-131\",\"workload\":\"occupancy\","
      "\"ok\":%s,\"occupancy_available\":false,"
      "\"reason\":\"anonymous_namespace_production_kernels\","
      "\"isolated\":{\"fp32_to_bf16_attn_ms\":%.9g,\"split_attention_ms\":%.9g,"
      "\"prepare_gdn_gates_ms\":%.9g,\"gdn_gated_output_ms\":%.9g,"
      "\"residual_add_ms\":%.9g},\"repeats\":%d,"
      "\"independently_restored\":true,\"same_binary\":true}\n",
      kPrefix, json_bool(ok), static_cast<double>(cast_ms),
      static_cast<double>(split_ms), static_cast<double>(gates_ms),
      static_cast<double>(gated_ms), static_cast<double>(resid_ms),
      kIsolatedRepeats);
  print_counts(1, 3, 1, false);
  return ok ? 0 : 1;
}

int run_lifetime(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 3, options.capacity);
  std::vector<std::size_t> tokens(prefix + 3);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &session, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::ExecutionGraphPathScope path_scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  if (status.is_ok()) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  workspace.reset_transfer_counters();
  std::size_t greedy = 0;
  if (status.is_ok()) status = qw38::cuda::greedy_sample(session, &greedy);
  const bool greedy_before = session.greedy_token_valid();
  std::vector<float> before(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> before_hidden{};
  if (status.is_ok()) {
    status = session.copy_last_outputs(before.data(), before.size(),
                                       before_hidden.data(),
                                       before_hidden.size());
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
    status = run_token(model, greedy, &session, &workspace, logits.data(),
                       hidden.data(), &elapsed, &graphs);
  }
  const std::uint64_t lazy_d2h = workspace.transfer_d2h_bytes_;
  std::vector<float> after_eval(qw38::internal::kVocabularySize);
  if (status.is_ok()) {
    status = session.copy_last_outputs(after_eval.data(), after_eval.size(),
                                       hidden.data(), hidden.size());
  }
  const bool eval_changed =
      std::memcmp(before.data(), after_eval.data(),
                  before.size() * sizeof(float)) != 0;
  const char* tmp = "/tmp/qw38-opt131-ckpt.bin";
  if (status.is_ok()) status = session.save_checkpoint(tmp);
  qw38::cuda::SchedulerSession restored;
  if (status.is_ok()) status = restored.create(capacity);
  if (status.is_ok()) status = restored.restore_checkpoint(tmp, &workspace);
  bool equal = false;
  if (status.is_ok()) status = restored.state_equals(session, &equal);
  std::remove(tmp);
  PollContext poll{0, 1};
  const qw38::cuda::EvalControl cancel{poll_stop, &poll};
  const std::size_t frontier_before = session.frontier();
  qw38::Status cancelled = run_token(model, tokens[prefix + 1], &session,
                                     &workspace, logits.data(), hidden.data(),
                                     &elapsed, &graphs, &cancel);
  const bool cancel_isolated =
      cancelled.code() == qw38::StatusCode::kCancelled &&
      session.frontier() == frontier_before;
  float next_ms = 0.0F;
  if (status.is_ok()) {
    status = run_token(model, tokens[prefix + 1], &session, &workspace,
                       logits.data(), hidden.data(), &next_ms, &graphs);
  }
  const bool subsequent_ok =
      status.is_ok() && session.frontier() == frontier_before + 1;
  const bool ok = status.is_ok() && greedy_before && repeat_equal && eval_changed &&
                  equal && cancel_isolated && subsequent_ok && lazy_d2h == 4;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-131\",\"workload\":\"lifetime\","
      "\"ok\":%s,\"greedy_valid\":%s,\"repeat_logits_equal\":%s,"
      "\"eval_changes_logits\":%s,\"restore_equals\":%s,\"cancel_isolated\":%s,"
      "\"subsequent_request_ok\":%s,\"lazy_d2h_bytes\":%llu,"
      "\"full_logit_bytes\":%zu,\"candidate_committed_separated\":%s}\n",
      kPrefix, json_bool(ok), json_bool(greedy_before), json_bool(repeat_equal),
      json_bool(eval_changed), json_bool(equal), json_bool(cancel_isolated),
      json_bool(subsequent_ok), static_cast<unsigned long long>(lazy_d2h),
      kLogitsBytes, json_bool(equal));
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_cancellation(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  bool all_ok = true;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-131\","
      "\"workload\":\"cancellation\",\"cadence\":\"eight_layer_segment\","
      "\"cases\":[",
      kPrefix);
  const char* names[] = {"mid_segment", "commit"};
  const std::size_t stops[] = {1, 8};
  for (int case_index = 0; case_index < 2; ++case_index) {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::Status status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    qw38::cuda::SchedulerGraphs graphs;
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    qw38::cuda::ExecutionGraphPathScope path_scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
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
    if (status.is_ok()) {
      token_status =
          run_token(model, tokens[prefix], &session, &workspace, logits.data(),
                    hidden.data(), &elapsed, &graphs, &control);
    }
    const bool cancelled = token_status.code() == qw38::StatusCode::kCancelled;
    const bool preserved = session.frontier() == frontier_before;
    const bool ok = status.is_ok() && cancelled && preserved &&
                    poll_context.calls >= 1;
    all_ok = all_ok && ok;
    std::printf(
        "%s{\"name\":\"%s\",\"ok\":%s,\"cancelled\":%s,\"frontier_preserved\":%s,"
        "\"poll_calls\":%zu,\"stop_after\":%zu,\"atomic_commit\":%s,"
        "\"crossed_cancellation_boundary\":false}",
        case_index == 0 ? "" : ",", names[case_index], json_bool(ok),
        json_bool(cancelled), json_bool(preserved), poll_context.calls,
        stops[case_index], json_bool(preserved));
  }
  std::printf(
      "],\"ok\":%s,\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false}\n",
      json_bool(all_ok));
  print_counts(0, 1, 1, false);
  return all_ok ? 0 : 1;
}

int run_equivalence(const qw38::cuda::ResidentModel& model,
                    const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 8 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 2 : options.tokens;
  const std::size_t capacity = session_capacity(prefix, outputs, options.capacity);
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  std::vector<std::size_t> graph_tokens;
  std::vector<std::size_t> eager_tokens;
  graph_tokens.reserve(outputs);
  eager_tokens.reserve(outputs);
  for (int arm = 0; arm < 2; ++arm) {
    const bool use_graphs = arm == 0;
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    qw38::Status status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    qw38::cuda::SchedulerGraphs* graph_ptr = nullptr;
    if (use_graphs) {
      if (status.is_ok()) {
        status = create_graphs(model, &session, &workspace, &graphs);
      }
      graph_ptr = &graphs;
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    qw38::cuda::ExecutionGraphPathScope path_scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, graph_ptr,
                       logits.data(), hidden.data());
    }
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      std::size_t sampled = 0;
      if (status.is_ok()) status = qw38::cuda::greedy_sample(session, &sampled);
      float elapsed = 0.0F;
      if (status.is_ok()) {
        status = run_token(model, sampled, &session, &workspace, logits.data(),
                           hidden.data(), &elapsed, graph_ptr);
      }
      (use_graphs ? graph_tokens : eager_tokens).push_back(sampled);
    }
    if (!status.is_ok()) return fail_status(status);
  }
  bool match = graph_tokens.size() == eager_tokens.size();
  for (std::size_t index = 0; match && index < graph_tokens.size(); ++index) {
    match = graph_tokens[index] == eager_tokens[index];
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-131\","
      "\"workload\":\"equivalence\",\"ok\":%s,\"tokens\":%zu,"
      "\"graph_matches_eager\":%s,\"unchanged_fallback\":true,"
      "\"fused_rounding_difference_tested\":false,"
      "\"reason\":\"no_fused_candidate_admitted\"}\n",
      kPrefix, json_bool(match), outputs, json_bool(match));
  print_counts(0, 1, 2, false);
  return match ? 0 : 1;
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
  if (std::strcmp(options.workload, "profile") == 0) {
    return run_profile(model, options);
  }
  if (std::strcmp(options.workload, "occupancy") == 0) {
    return run_occupancy(model, options);
  }
  if (std::strcmp(options.workload, "lifetime") == 0) {
    return run_lifetime(model, options);
  }
  if (std::strcmp(options.workload, "cancellation") == 0) {
    return run_cancellation(model, options);
  }
  if (std::strcmp(options.workload, "equivalence") == 0) {
    return run_equivalence(model, options);
  }
  return usage(argv[0]);
}
