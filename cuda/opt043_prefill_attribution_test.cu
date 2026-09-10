#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#include "full_scheduler.h"
#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr std::size_t kCapacity = 131072;
constexpr std::size_t kPromptTokens = 4096;
constexpr float kAbsTolMs = 0.05F;
constexpr float kRelTol = 1.0e-4F;
constexpr char kPrefix[] = "QW38_OPT043_PREFILL_ATTRIBUTION_RESULT=";

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

bool category_measured(const qw38::cuda::TimingValue& value) {
  return value.measured;
}

bool timings_close(float left, float right) {
  const float scale = std::max(std::fabs(left), std::fabs(right));
  return std::fabs(left - right) <= std::max(kAbsTolMs, kRelTol * scale);
}

void print_timing(const char* name, const qw38::cuda::TimingValue& value,
                  bool last) {
  std::printf("\"%s\":{\"ms\":%.9g,\"measured\":%s}%s", name, value.milliseconds,
              value.measured ? "true" : "false", last ? "" : ",");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
    return 2;
  }
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(argv[1], &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(argv[1]);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  qw38::cuda::ResidentModel model;
  if (status.is_ok()) {
    status = model.upload(weights, mapping.data(), mapping.size());
  }
  qw38::cuda::SchedulerSession session;
  if (status.is_ok()) status = session.create(kCapacity);
  qw38::cuda::SchedulerWorkspace workspace;
  if (status.is_ok()) status = workspace.create(kCapacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return fail_status(status);

  std::vector<std::size_t> tokens(kPromptTokens);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult result{};
  qw38::cuda::PrefillAttribution attribution;
  attribution.record_leaves = true;
  const auto host_started = std::chrono::steady_clock::now();
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), tokens.size(), &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &result, nullptr, &graphs,
      qw38::cuda::GdnScanPath::kFusedTokenLoop, &attribution);
  const float raw_host_wall_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - host_started)
          .count());
  if (!status.is_ok()) return fail_status(status);

  const bool measured =
      category_measured(attribution.embedding) &&
      category_measured(attribution.mixer_mmq) &&
      category_measured(attribution.gdn_core) &&
      category_measured(attribution.attention_core) &&
      category_measured(attribution.ffn_mmq) &&
      category_measured(attribution.logits) &&
      category_measured(attribution.commit_sync) &&
      category_measured(attribution.graph) &&
      category_measured(attribution.other_idle) &&
      category_measured(attribution.wall);
  if (!measured || attribution.prompt_tokens != kPromptTokens ||
      attribution.evaluated_tokens != kPromptTokens ||
      attribution.chunk_count != 1 ||
      attribution.prompt_graph_launches != 64 ||
      attribution.mixer_mmq.milliseconds <= 0.0F ||
      attribution.gdn_core.milliseconds <= 0.0F ||
      attribution.attention_core.milliseconds <= 0.0F ||
      attribution.ffn_mmq.milliseconds <= 0.0F ||
      attribution.graph.milliseconds <= 0.0F ||
      !attribution.record_leaves ||
      graphs.prompt_graph_rows() != qw38::cuda::kPromptChunkRows ||
      session.frontier() != kPromptTokens) {
    std::fprintf(stderr, "OPT-043 4K prefill leaf attribution is incomplete\n");
    return 1;
  }

  const float gpu_event_sum_ms =
      attribution.embedding.milliseconds + attribution.mixer_mmq.milliseconds +
      attribution.gdn_core.milliseconds +
      attribution.attention_core.milliseconds +
      attribution.ffn_mmq.milliseconds + attribution.logits.milliseconds +
      attribution.commit_sync.milliseconds;
  const float graph_host_interval_ms = attribution.graph.milliseconds;
  const float attributed_sum_ms =
      gpu_event_sum_ms + graph_host_interval_ms +
      attribution.other_idle.milliseconds;
  const float adjusted_reconstruction_ms = attribution.wall.milliseconds;
  const bool wall_raised =
      adjusted_reconstruction_ms > raw_host_wall_ms &&
      !timings_close(adjusted_reconstruction_ms, raw_host_wall_ms);
  const float tok_s =
      adjusted_reconstruction_ms > 0.0F
          ? static_cast<float>(kPromptTokens) * 1000.0F /
                adjusted_reconstruction_ms
          : 0.0F;

  cudaDeviceProp prop{};
  int device = 0;
  int driver = 0;
  int runtime = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);
  const qw38::cuda::LeafTimings& leaves = attribution.leaves;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-043\",\"status\":\"measured\","
      "\"mode\":\"prefill_leaves\",\"device\":\"%s\","
      "\"compute_capability\":\"%d.%d\",\"driver\":\"%d.%d\","
      "\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\","
      "\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\","
      "\"prompt_tokens\":%zu,\"evaluated_tokens\":%zu,\"chunk_count\":%zu,"
      "\"prompt_graph_launches\":%u,\"cold\":true,\"cache_policy\":\"disabled\","
      "\"graphs_created\":true,\"record_leaves\":true,"
      "\"categories_ms\":{\"embedding\":%.9g,\"mixer_mmq\":%.9g,\"gdn_core\":%.9g,"
      "\"attention_core\":%.9g,\"ffn_mmq\":%.9g,\"logits\":%.9g,"
      "\"commit_sync\":%.9g,\"graph\":%.9g,\"other_idle\":%.9g},"
      "\"attributed_sum_ms\":%.9g,\"wall_ms\":%.9g,\"tok_s\":%.9g,"
      "\"raw_host_wall_ms\":%.9g,\"gpu_event_sum_ms\":%.9g,"
      "\"graph_host_interval_ms\":%.9g,\"adjusted_reconstruction_ms\":%.9g,"
      "\"wall_raised\":%s,\"fused\":{"
      "\"gdn_conv_fused_with_recurrence\":%s,"
      "\"gdn_output_norm_fused_with_gate\":%s,"
      "\"attention_prep_fused_with_core\":%s,"
      "\"ffn_graph_fused\":%s,"
      "\"d2h_overlaps_state_copies\":%s},"
      "\"leaves\":{",
      kPrefix, prop.name, prop.major, prop.minor, driver / 1000,
      (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10, utc,
      attribution.prompt_tokens, attribution.evaluated_tokens,
      attribution.chunk_count, attribution.prompt_graph_launches,
      attribution.embedding.milliseconds, attribution.mixer_mmq.milliseconds,
      attribution.gdn_core.milliseconds, attribution.attention_core.milliseconds,
      attribution.ffn_mmq.milliseconds, attribution.logits.milliseconds,
      attribution.commit_sync.milliseconds, attribution.graph.milliseconds,
      attribution.other_idle.milliseconds, attributed_sum_ms,
      adjusted_reconstruction_ms, tok_s, raw_host_wall_ms, gpu_event_sum_ms,
      graph_host_interval_ms, adjusted_reconstruction_ms,
      wall_raised ? "true" : "false",
      leaves.gdn_conv_fused_with_recurrence ? "true" : "false",
      leaves.gdn_output_norm_fused_with_gate ? "true" : "false",
      leaves.attention_prep_fused_with_core ? "true" : "false",
      leaves.ffn_graph_fused ? "true" : "false",
      leaves.d2h_overlaps_state_copies ? "true" : "false");
  print_timing("embedding", leaves.embedding, false);
  print_timing("input_norm", leaves.input_norm, false);
  print_timing("activation_staging_mixer", leaves.activation_staging_mixer,
               false);
  print_timing("proj_packed_qkv", leaves.proj_packed_qkv, false);
  print_timing("proj_value_gate", leaves.proj_value_gate, false);
  print_timing("proj_alpha", leaves.proj_alpha, false);
  print_timing("proj_beta", leaves.proj_beta, false);
  print_timing("proj_gdn_output", leaves.proj_gdn_output, false);
  print_timing("proj_query_gate", leaves.proj_query_gate, false);
  print_timing("proj_key", leaves.proj_key, false);
  print_timing("proj_value", leaves.proj_value, false);
  print_timing("proj_attn_output", leaves.proj_attn_output, false);
  print_timing("gdn_conv_qk_norm_recurrence",
               leaves.gdn_conv_qk_norm_recurrence, false);
  print_timing("attn_qk_prep_softmax_pv_merge",
               leaves.attn_qk_prep_softmax_pv_merge, false);
  print_timing("ffn_norm", leaves.ffn_norm, false);
  print_timing("activation_staging_ffn", leaves.activation_staging_ffn, false);
  print_timing("proj_ffn_gate", leaves.proj_ffn_gate, false);
  print_timing("proj_ffn_up", leaves.proj_ffn_up, false);
  print_timing("swiglu", leaves.swiglu, false);
  print_timing("proj_ffn_down", leaves.proj_ffn_down, false);
  print_timing("residual_ffn", leaves.residual_ffn, false);
  print_timing("logits_norm", leaves.logits_norm, false);
  print_timing("logits_projection", leaves.logits_projection, false);
  print_timing("d2h", leaves.d2h, false);
  print_timing("state_copies", leaves.state_copies, false);
  print_timing("host_graph_submit", leaves.host_graph_submit, true);
  std::printf("}}\n");
  std::printf("status=passed\n");
  return 0;
}
