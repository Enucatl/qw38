#include <array>
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
constexpr std::size_t kPromptTokens = 2048;

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

bool category_measured(const qw38::cuda::TimingValue& value) {
  return value.measured;
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
  // Production prompt GDN path (warp-column quality). Overlay remains the
  // OPT-013 alternate and is not used for OPT-019 core-after attribution.
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), tokens.size(), &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &result, nullptr, &graphs,
      qw38::cuda::GdnScanPath::kFusedTokenLoop, &attribution);
  if (!status.is_ok()) return fail_status(status);

  const bool measured =
      category_measured(attribution.embedding) &&
      category_measured(attribution.gdn) &&
      category_measured(attribution.attention) &&
      category_measured(attribution.ffn_mmq) &&
      category_measured(attribution.logits) &&
      category_measured(attribution.commit_sync) &&
      category_measured(attribution.graph) &&
      category_measured(attribution.other_idle) &&
      category_measured(attribution.wall);
  if (!measured || attribution.prompt_tokens != kPromptTokens ||
      attribution.evaluated_tokens != kPromptTokens ||
      attribution.chunk_count != 1 ||
      attribution.prompt_graph_launches != 0 ||
      graphs.prompt_graph_rows() != qw38::cuda::kPromptChunkRows ||
      session.frontier() != kPromptTokens) {
    std::fprintf(stderr, "prefill attribution record is incomplete\n");
    return 1;
  }

  const float attributed =
      attribution.embedding.milliseconds + attribution.gdn.milliseconds +
      attribution.attention.milliseconds + attribution.ffn_mmq.milliseconds +
      attribution.logits.milliseconds + attribution.commit_sync.milliseconds +
      attribution.graph.milliseconds + attribution.other_idle.milliseconds;
  const float wall_ms = attribution.wall.milliseconds;
  const float tok_s =
      wall_ms > 0.0F ? static_cast<float>(kPromptTokens) * 1000.0F / wall_ms
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
  std::printf(
      "QW38_PREFILL_ATTRIBUTION_RESULT={\"schema_version\":1,\"task\":\"OPT-014\","
      "\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\","
      "\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\","
      "\"prompt_tokens\":%zu,\"evaluated_tokens\":%zu,\"chunk_count\":%zu,"
      "\"prompt_graph_launches\":%u,\"cold\":true,\"cache_policy\":\"disabled\","
      "\"categories_ms\":{\"embedding\":%.9g,\"gdn\":%.9g,\"attention\":%.9g,"
      "\"ffn_mmq\":%.9g,\"logits\":%.9g,\"commit_sync\":%.9g,\"graph\":%.9g,"
      "\"other_idle\":%.9g},\"attributed_sum_ms\":%.9g,\"wall_ms\":%.9g,"
      "\"tok_s\":%.9g,\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\","
      "\"proof_limit\":\"live CUDA-event attribution; 2048-token cold prefill; "
      "no Nsight capture; not a throughput gate; not llama.cpp parity\"}\n",
      prop.name, prop.major, prop.minor, driver / 1000, (driver % 1000) / 10,
      runtime / 1000, (runtime % 1000) / 10, utc, attribution.prompt_tokens,
      attribution.evaluated_tokens, attribution.chunk_count,
      attribution.prompt_graph_launches, attribution.embedding.milliseconds,
      attribution.gdn.milliseconds, attribution.attention.milliseconds,
      attribution.ffn_mmq.milliseconds, attribution.logits.milliseconds,
      attribution.commit_sync.milliseconds, attribution.graph.milliseconds,
      attribution.other_idle.milliseconds, attributed, wall_ms, tok_s);
  std::printf("status=passed\n");
  return 0;
}
