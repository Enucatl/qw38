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
constexpr std::size_t kPromptTokens = 4096;
constexpr int kReplicates = 3;

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
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
  if (!status.is_ok()) return fail_status(status);

  std::vector<std::size_t> tokens(kPromptTokens);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);

  float wall_ms[kReplicates]{};
  float tok_s[kReplicates]{};
  for (int replicate = 0; replicate < kReplicates; ++replicate) {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    status = session.create(kCapacity);
    if (status.is_ok()) status = workspace.create(kCapacity);
    if (status.is_ok()) status = graphs.create(model, &workspace);
    if (!status.is_ok()) return fail_status(status);
    if (graphs.prompt_graph_rows() != qw38::cuda::kPromptChunkRows ||
        graphs.prompt_graph_count() != qw38::internal::kModelLayerCount) {
      std::fprintf(stderr, "4K FFN prompt graphs are incomplete\n");
      return 1;
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    qw38::cuda::SyncResult result{};
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    if (error == cudaSuccess) error = cudaEventRecord(start);
    if (error != cudaSuccess) {
      std::fprintf(stderr, "cudaEvent: %s\n", cudaGetErrorString(error));
      return 1;
    }
    status = qw38::cuda::sync_tokens(
        model, tokens.data(), tokens.size(), &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &result, nullptr, &graphs);
    if (!status.is_ok()) return fail_status(status);
    error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(&wall_ms[replicate], start, stop);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (error != cudaSuccess) {
      std::fprintf(stderr, "timing: %s\n", cudaGetErrorString(error));
      return 1;
    }
    tok_s[replicate] =
        wall_ms[replicate] > 0.0F
            ? static_cast<float>(kPromptTokens) * 1000.0F / wall_ms[replicate]
            : 0.0F;
    if (session.frontier() != kPromptTokens) {
      std::fprintf(stderr, "4K FFN session is incomplete\n");
      return 1;
    }
  }

  float mean_tok_s = 0.0F;
  for (int replicate = 0; replicate < kReplicates; ++replicate) {
    mean_tok_s += tok_s[replicate];
  }
  mean_tok_s /= static_cast<float>(kReplicates);

  std::printf(
      "QW38_PREFILL_4K_FFN_RESULT={\"schema_version\":1,\"task\":\"OPT-025\","
      "\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"measurement_utc\":\"%s\",\"prompt_tokens\":%zu,\"replicates\":%d,"
      "\"wall_ms\":[%.9g,%.9g,%.9g],\"tok_s\":[%.9g,%.9g,%.9g],"
      "\"mean_tok_s\":%.9g,\"cold\":true,\"cache_policy\":\"disabled\","
      "\"attribution\":null,\"graphs_created\":true,\"prompt_graph_rows\":%zu,"
      "\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\"}\n",
      prop.name, prop.major, prop.minor, utc, kPromptTokens, kReplicates,
      wall_ms[0], wall_ms[1], wall_ms[2], tok_s[0], tok_s[1], tok_s[2],
      mean_tok_s, kPromptTokens);
  std::printf("status=passed\n");
  return 0;
}
