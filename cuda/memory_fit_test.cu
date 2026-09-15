#include "full_scheduler.h"
#include "attention_decode.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <cuda_runtime.h>

#include "mixer.h"
#include "model.h"
#include "weights.h"

namespace {

constexpr std::size_t kCapacity = 131072;
constexpr std::size_t kGdnLayers = 48;
constexpr std::size_t kAttentionLayers = 16;
constexpr std::size_t kReserveBytes = 1536ULL * 1024 * 1024;

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

std::size_t resident_host_bytes() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      std::size_t kibibytes = 0;
      status >> kibibytes;
      return kibibytes * 1024;
    }
    std::string rest;
    std::getline(status, rest);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  const char* attention_pipeline = nullptr;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--attention-pipeline") == 0 &&
        index + 1 < argc) {
      attention_pipeline = argv[++index];
    } else if (model_path == nullptr) {
      model_path = argv[index];
    } else {
      std::fprintf(stderr,
                   "usage: qw38-cuda-memory-fit-test MODEL "
                   "[--attention-pipeline PATH]\n");
      return 1;
    }
  }
  if (model_path == nullptr) {
    std::fprintf(stderr,
                 "usage: qw38-cuda-memory-fit-test MODEL "
                 "[--attention-pipeline PATH]\n");
    return 1;
  }
  if (attention_pipeline != nullptr &&
      !qw38::cuda::apply_attention_pipeline_ident(attention_pipeline)) {
    std::fprintf(stderr, "invalid --attention-pipeline %s\n",
                 attention_pipeline);
    return 1;
  }
  cudaError_t error = cudaFree(nullptr);
  std::size_t free_baseline = 0;
  std::size_t total = 0;
  if (error == cudaSuccess) error = cudaMemGetInfo(&free_baseline, &total);
  if (error != cudaSuccess) {
    std::fprintf(stderr, "cannot establish CUDA memory baseline\n");
    return 1;
  }

  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (!status.is_ok()) return fail_status(status);

  qw38::cuda::ResidentModel model;
  status = model.upload(weights, mapping.data(), mapping.size());
  std::size_t free_after_model = 0;
  if (status.is_ok() &&
      cudaMemGetInfo(&free_after_model, &total) != cudaSuccess) {
    status = {qw38::StatusCode::kInternal,
              "cannot measure memory after model upload"};
  }
  qw38::cuda::SchedulerSession session;
  if (status.is_ok()) status = session.create(kCapacity);
  std::size_t free_after_session = 0;
  if (status.is_ok() &&
      cudaMemGetInfo(&free_after_session, &total) != cudaSuccess) {
    status = {qw38::StatusCode::kInternal,
              "cannot measure memory after session allocation"};
  }
  qw38::cuda::SchedulerWorkspace workspace;
  if (status.is_ok()) status = workspace.create(kCapacity);
  std::size_t free_after_workspace = 0;
  if (status.is_ok() &&
      cudaMemGetInfo(&free_after_workspace, &total) != cudaSuccess) {
    status = {qw38::StatusCode::kInternal,
              "cannot measure memory after workspace allocation"};
  }
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &workspace, &session);
  std::size_t free_after_graphs = 0;
  if (status.is_ok() &&
      cudaMemGetInfo(&free_after_graphs, &total) != cudaSuccess) {
    status = {qw38::StatusCode::kInternal,
              "cannot measure memory after graph creation"};
  }
  if (!status.is_ok()) return fail_status(status);

  const std::size_t gdn_bytes =
      kGdnLayers *
      (qw38::internal::kGdnConvolutionValues +
       qw38::internal::kGdnRecurrentStateValues) *
      sizeof(float);
  const std::size_t kv_bytes =
      kAttentionLayers * kCapacity * qw38::internal::kAttentionKvWidth *
      sizeof(__nv_bfloat16) * 2;
  const std::size_t graph_bytes = graphs.allocated_bytes();
  const std::size_t graph_measured =
      free_after_workspace >= free_after_graphs
          ? free_after_workspace - free_after_graphs
          : 0;
  const std::size_t expected_decode_segments =
      static_cast<std::size_t>(
          qw38::cuda::effective_decode_graph_topology_count()) *
      qw38::cuda::kDecodeSegmentCount;
  const bool segment_path = qw38::cuda::execution_graph_uses_decode_segments8();
  const bool graphs_ok =
      graph_bytes > 0 && graphs.prompt_graph_count() == 64 &&
      graphs.prompt_graph_rows() == 4096 &&
      (segment_path
           ? (graphs.decode_graph_count() == 0 &&
              graphs.decode_segment_graph_count() == expected_decode_segments)
           : (graphs.decode_graph_count() == 64 &&
              graphs.graph_count() == 128 &&
              graphs.decode_segment_graph_count() == 0));
  const std::size_t explicit_bytes =
      model.resident_bytes() + session.allocated_bytes() +
      workspace.allocated_bytes() + graph_bytes;
  const std::size_t measured_delta = free_baseline - free_after_graphs;
  const std::size_t allocator_delta =
      measured_delta >= explicit_bytes ? measured_delta - explicit_bytes : 0;
  const std::size_t runtime_bytes = total - free_baseline;
  const std::size_t host_rss = resident_host_bytes();
  // Session now owns GDN+KV plus decode launch-state / logits device buffers
  // (OPT-117+). Historical MEM-001 equality was GDN+KV only.
  const bool arithmetic =
      session.allocated_bytes() >= gdn_bytes + kv_bytes &&
      graph_bytes == graph_measured && graphs_ok &&
      model.resident_bytes() == 18973870432ULL &&
      workspace.allocated_bytes() == 1831836288ULL;
  const bool passed = arithmetic && session.capacity() == kCapacity &&
                      free_after_graphs >= kReserveBytes;

  std::printf("memory_owner=runtime_context bytes=%zu\n", runtime_bytes);
  std::printf("memory_owner=resident_model bytes=%zu delta=%zu\n",
              model.resident_bytes(), free_baseline - free_after_model);
  std::printf("memory_owner=gdn_state bytes=%zu\n", gdn_bytes);
  std::printf("memory_owner=attention_kv bytes=%zu\n", kv_bytes);
  std::printf("memory_owner=session_total bytes=%zu delta=%zu\n",
              session.allocated_bytes(), free_after_model - free_after_session);
  std::printf("memory_owner=workspace bytes=%zu delta=%zu\n",
              workspace.allocated_bytes(),
              free_after_session - free_after_workspace);
  std::printf("memory_owner=allocator_delta bytes=%zu\n", allocator_delta);
  std::printf("memory_owner=graphs bytes=%zu count=%zu decode=%zu prompt=%zu "
              "decode_segments=%zu prompt_rows=%zu graph_measured=%zu\n",
              graph_bytes, graphs.graph_count(), graphs.decode_graph_count(),
              graphs.prompt_graph_count(), graphs.decode_segment_graph_count(),
              graphs.prompt_graph_rows(), graph_measured);
  std::printf("memory_host=rss bytes=%zu\n", host_rss);
  std::printf("memory_fit=post_graph capacity=%zu explicit_bytes=%zu "
              "measured_delta=%zu free_bytes=%zu reserve_required=%zu "
              "total_bytes=%zu arithmetic=%s passed=%s\n",
              kCapacity, explicit_bytes, measured_delta, free_after_graphs,
              kReserveBytes, total, arithmetic ? "true" : "false",
              passed ? "true" : "false");
  std::printf("memory_admission=post_graph passed=%s\n",
              passed ? "true" : "false");
  std::printf("status=%s\n", passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
