#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "full_scheduler.h"
#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr std::size_t kCapacity = 131072;
constexpr char kPrefix[] = "QW38_DECODE_ATTRIBUTION_RESULT=";

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

bool category_measured(const qw38::cuda::TimingValue& value) {
  return value.measured;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf PREFIX\n", argv[0]);
    return 2;
  }
  const std::size_t prefix = static_cast<std::size_t>(std::atoi(argv[2]));
  if (prefix != 128 && prefix != 2048) {
    std::fprintf(stderr, "PREFIX must be 128 or 2048\n");
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

  std::vector<std::size_t> tokens(prefix + 1);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult result{};
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), prefix, &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &result, nullptr, &graphs);
  if (!status.is_ok()) return fail_status(status);

  float elapsed_ms = 0.0F;
  qw38::cuda::DecodeAttribution attribution;
  status = qw38::cuda::execute_token(
      model, tokens[prefix], &session, &workspace, logits.data(), logits.size(),
      hidden.data(), hidden.size(), &elapsed_ms, nullptr, nullptr,
      qw38::cuda::PointwisePath::kFused, &graphs, &attribution);
  if (!status.is_ok()) return fail_status(status);

  const bool measured =
      category_measured(attribution.embedding) &&
      category_measured(attribution.mixer_mmv) &&
      category_measured(attribution.gdn_core) &&
      category_measured(attribution.attention_core) &&
      category_measured(attribution.ffn_mmv) &&
      category_measured(attribution.logits) &&
      category_measured(attribution.state_commit) &&
      category_measured(attribution.graph) &&
      category_measured(attribution.other_idle) &&
      category_measured(attribution.wall);
  if (!measured || attribution.mixer_mmv.milliseconds <= 0.0F ||
      attribution.gdn_core.milliseconds <= 0.0F ||
      attribution.attention_core.milliseconds <= 0.0F ||
      attribution.ffn_mmv.milliseconds <= 0.0F ||
      session.frontier() != prefix + 1) {
    std::fprintf(stderr, "decode attribution record is incomplete\n");
    return 1;
  }

  const float attributed =
      attribution.embedding.milliseconds + attribution.mixer_mmv.milliseconds +
      attribution.gdn_core.milliseconds +
      attribution.attention_core.milliseconds +
      attribution.ffn_mmv.milliseconds + attribution.logits.milliseconds +
      attribution.state_commit.milliseconds + attribution.graph.milliseconds +
      attribution.other_idle.milliseconds;
  const float wall_ms = attribution.wall.milliseconds;

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
      "%s{\"schema_version\":1,\"task\":\"OPT-032\",\"status\":\"measured\","
      "\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\","
      "\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\","
      "\"prefix\":%zu,\"graphs_created\":true,"
      "\"categories_ms\":{\"embedding\":%.9g,\"mixer_mmv\":%.9g,\"gdn_core\":%.9g,"
      "\"attention_core\":%.9g,\"ffn_mmv\":%.9g,\"logits\":%.9g,"
      "\"state_commit\":%.9g,\"graph\":%.9g,\"other_idle\":%.9g},"
      "\"attributed_sum_ms\":%.9g,\"wall_ms\":%.9g,"
      "\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\"}\n",
      kPrefix, prop.name, prop.major, prop.minor, driver / 1000,
      (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10, utc, prefix,
      attribution.embedding.milliseconds, attribution.mixer_mmv.milliseconds,
      attribution.gdn_core.milliseconds, attribution.attention_core.milliseconds,
      attribution.ffn_mmv.milliseconds, attribution.logits.milliseconds,
      attribution.state_commit.milliseconds, attribution.graph.milliseconds,
      attribution.other_idle.milliseconds, attributed, wall_ms);
  std::printf("status=passed\n");
  return 0;
}
