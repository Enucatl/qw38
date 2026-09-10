#include "full_scheduler.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr int kThreads = 256;
constexpr std::size_t kWideRows = 4096;
constexpr std::size_t kFallbackRows = 64;
constexpr std::size_t kSmallCapacity = 65;

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
};

struct KernelGraph final {
  bool ok = false;
  int kernel_nodes = 0;
  bool has_rowwise = false;
  dim3 grid{};
  dim3 block{};
};

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

qw38::Status poll(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls == context->stop_after) {
    return {qw38::StatusCode::kCancelled, "prompt graph test cancellation"};
  }
  return qw38::Status::ok();
}

bool outputs_equal(
    const std::vector<float>& left_logits, const std::vector<float>& right_logits,
    const std::array<float, qw38::internal::kResidualWidth>& left_hidden,
    const std::array<float, qw38::internal::kResidualWidth>& right_hidden) {
  return std::memcmp(left_logits.data(), right_logits.data(),
                     left_logits.size() * sizeof(float)) == 0 &&
         std::memcmp(left_hidden.data(), right_hidden.data(),
                     left_hidden.size() * sizeof(float)) == 0;
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

void fill_tokens(std::size_t* tokens, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

template <typename Fn>
KernelGraph capture(Fn launch) {
  KernelGraph info;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaError_t error =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  }
  if (error == cudaSuccess) error = launch(stream);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  std::size_t count = 0;
  if (error == cudaSuccess) error = cudaGraphGetNodes(graph, nullptr, &count);
  std::vector<cudaGraphNode_t> nodes(count);
  if (error == cudaSuccess) {
    error = cudaGraphGetNodes(graph, nodes.data(), &count);
  }
  for (cudaGraphNode_t node : nodes) {
    if (error != cudaSuccess) break;
    cudaGraphNodeType type{};
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
      error = cudaErrorInvalidValue;
      break;
    }
    if (type != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams params{};
    if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
      error = cudaErrorInvalidValue;
      break;
    }
    ++info.kernel_nodes;
    const unsigned int norm_threads = static_cast<unsigned int>(
        qw38::cuda::effective_rms_norm_threads());
    if (params.gridDim.x == kWideRows &&
        (params.blockDim.x == kThreads || params.blockDim.x == norm_threads ||
         params.blockDim.x == 128U || params.blockDim.x == 256U)) {
      info.has_rowwise = true;
      info.grid = params.gridDim;
      info.block = params.blockDim;
    }
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  info.ok = error == cudaSuccess;
  return info;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qw38-cuda-prompt-graph-test MODEL\n");
    return 1;
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
  if (status.is_ok()) status = model.upload(weights, mapping.data(), mapping.size());
  if (!status.is_ok()) return fail_status(status);

  qw38::cuda::SchedulerSession graph_session;
  qw38::cuda::SchedulerWorkspace graph_workspace;
  status = graph_session.create(kWideRows);
  if (status.is_ok()) status = graph_workspace.create(kWideRows);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &graph_workspace);
  if (!status.is_ok()) return fail_status(status);
  const bool graphs_4096_ok =
      graphs.decode_graph_count() == 64 && graphs.prompt_graph_count() == 64 &&
      graphs.prompt_graph_rows() == kWideRows && graphs.graph_count() == 128 &&
      graphs.allocated_bytes() > 0;

  bool graphs_65_ok = false;
  {
    qw38::cuda::SchedulerSession small_session;
    qw38::cuda::SchedulerWorkspace small_workspace;
    qw38::cuda::SchedulerGraphs small_graphs;
    status = small_session.create(kSmallCapacity);
    if (status.is_ok()) status = small_workspace.create(kSmallCapacity);
    if (status.is_ok()) status = small_graphs.create(model, &small_workspace);
    if (!status.is_ok()) return fail_status(status);
    graphs_65_ok = small_graphs.decode_graph_count() == 64 &&
                   small_graphs.prompt_graph_count() == 0 &&
                   small_graphs.prompt_graph_rows() == 0 &&
                   small_graphs.graph_count() == 64;
  }

  const KernelGraph prompt_ffn = capture([&](cudaStream_t stream) {
    return qw38::cuda::execute_prompt_ffn(
        model.common_layer(0), graph_workspace.prompt_residual_a_,
        &graph_workspace, graph_workspace.prompt_residual_b_,
        graph_workspace.prompt_residual_a_, model.next_input_norm(0), kWideRows,
        stream);
  });
  const bool launch_ok =
      prompt_ffn.ok && prompt_ffn.kernel_nodes >= 6 && prompt_ffn.has_rowwise &&
      prompt_ffn.grid.x == kWideRows && prompt_ffn.block.x == kThreads;

  std::array<std::size_t, kWideRows> tokens{};
  fill_tokens(tokens.data(), tokens.size());
  std::vector<float> mismatch_logits(qw38::internal::kVocabularySize, 17.0F);
  std::array<float, qw38::internal::kResidualWidth> mismatch_hidden{};
  mismatch_hidden.fill(23.0F);
  qw38::cuda::SchedulerSession mismatch_session;
  qw38::cuda::SchedulerWorkspace mismatch_workspace;
  status = mismatch_session.create(kWideRows);
  if (status.is_ok()) status = mismatch_workspace.create(kWideRows);
  if (!status.is_ok()) return fail_status(status);
  const qw38::Status mismatch_status = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), kWideRows, &mismatch_session, &mismatch_workspace,
      mismatch_logits.data(), mismatch_logits.size(), mismatch_hidden.data(),
      mismatch_hidden.size(), nullptr,
      qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr, &graphs);
  const bool mismatch_ok =
      mismatch_status.code() == qw38::StatusCode::kInvalidArgument &&
      mismatch_session.frontier() == 0;

  std::vector<float> unfused_logits(qw38::internal::kVocabularySize, 5.0F);
  std::array<float, qw38::internal::kResidualWidth> unfused_hidden{};
  unfused_hidden.fill(9.0F);
  const qw38::Status unfused_status = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), kWideRows, &graph_session, &graph_workspace,
      unfused_logits.data(), unfused_logits.size(), unfused_hidden.data(),
      unfused_hidden.size(), nullptr,
      qw38::cuda::PromptPipelinePath::kUnfusedSerial, nullptr, &graphs);
  const bool unfused_ok =
      unfused_status.code() == qw38::StatusCode::kInvalidArgument &&
      graph_session.frontier() == 0;

  qw38::cuda::SchedulerSession ordinary_session;
  qw38::cuda::SchedulerWorkspace ordinary_workspace;
  status = ordinary_session.create(kWideRows);
  if (status.is_ok()) status = ordinary_workspace.create(kWideRows);
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::vector<float> ordinary_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  std::array<float, qw38::internal::kResidualWidth> ordinary_hidden{};
  qw38::cuda::PromptPipelineCounters graph_counters;
  qw38::cuda::PromptPipelineCounters ordinary_counters;
  cudaEvent_t graph_start = nullptr;
  cudaEvent_t graph_stop = nullptr;
  cudaEvent_t ordinary_start = nullptr;
  cudaEvent_t ordinary_stop = nullptr;
  cudaError_t event_error = cudaEventCreate(&graph_start);
  if (event_error == cudaSuccess) event_error = cudaEventCreate(&graph_stop);
  if (event_error == cudaSuccess) event_error = cudaEventCreate(&ordinary_start);
  if (event_error == cudaSuccess) event_error = cudaEventCreate(&ordinary_stop);
  if (event_error != cudaSuccess) {
    std::fprintf(stderr, "cannot create prompt graph timing events\n");
    return 1;
  }
  event_error = cudaEventRecord(graph_start, graph_workspace.prompt_compute_stream_);
  if (event_error != cudaSuccess) {
    std::fprintf(stderr, "graph timing start failed\n");
    return 1;
  }
  status = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), kWideRows, &graph_session, &graph_workspace,
      graph_logits.data(), graph_logits.size(), graph_hidden.data(),
      graph_hidden.size(), nullptr,
      qw38::cuda::PromptPipelinePath::kFusedOverlapped, &graph_counters, &graphs);
  if (!status.is_ok()) return fail_status(status);
  event_error = cudaEventRecord(graph_stop, graph_workspace.prompt_compute_stream_);
  if (event_error == cudaSuccess) event_error = cudaEventSynchronize(graph_stop);
  if (event_error != cudaSuccess) {
    std::fprintf(stderr, "graph timing stop failed\n");
    return 1;
  }
  event_error =
      cudaEventRecord(ordinary_start, ordinary_workspace.prompt_compute_stream_);
  if (event_error != cudaSuccess) {
    std::fprintf(stderr, "ordinary timing start failed\n");
    return 1;
  }
  status = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), kWideRows, &ordinary_session, &ordinary_workspace,
      ordinary_logits.data(), ordinary_logits.size(), ordinary_hidden.data(),
      ordinary_hidden.size(), nullptr,
      qw38::cuda::PromptPipelinePath::kFusedOverlapped, &ordinary_counters,
      nullptr);
  if (!status.is_ok()) return fail_status(status);
  event_error =
      cudaEventRecord(ordinary_stop, ordinary_workspace.prompt_compute_stream_);
  if (event_error == cudaSuccess) event_error = cudaEventSynchronize(ordinary_stop);
  if (event_error != cudaSuccess) {
    std::fprintf(stderr, "ordinary timing stop failed\n");
    return 1;
  }
  float graph_ms = 0.0F;
  float ordinary_ms = 0.0F;
  event_error = cudaEventElapsedTime(&graph_ms, graph_start, graph_stop);
  if (event_error == cudaSuccess) {
    event_error =
        cudaEventElapsedTime(&ordinary_ms, ordinary_start, ordinary_stop);
  }
  cudaEventDestroy(ordinary_stop);
  cudaEventDestroy(ordinary_start);
  cudaEventDestroy(graph_stop);
  cudaEventDestroy(graph_start);
  if (event_error != cudaSuccess) {
    std::fprintf(stderr, "prompt graph elapsed-time failed\n");
    return 1;
  }
  bool equality_state = false;
  status = graph_session.state_equals(ordinary_session, &equality_state);
  if (!status.is_ok()) return fail_status(status);
  const bool hidden_exact =
      std::memcmp(graph_hidden.data(), ordinary_hidden.data(),
                  graph_hidden.size() * sizeof(float)) == 0;
  const bool logits_exact =
      std::memcmp(graph_logits.data(), ordinary_logits.data(),
                  graph_logits.size() * sizeof(float)) == 0;
  const bool equality_ok = equality_state && hidden_exact && logits_exact &&
                           graph_session.frontier() == kWideRows &&
                           ordinary_session.frontier() == kWideRows;
  const std::size_t equality_frontier = graph_session.frontier();
  const bool graph_counts =
      graph_counters.prompt_graph_launches == 64 &&
      graph_counters.embedding_kernel_launches == 1 &&
      graph_counters.scatter_kernel_launches == 1 &&
      graph_counters.blocking_d2h_copies == 0 &&
      graph_counters.async_d2h_copies == 2 &&
      graph_counters.device_synchronizes == 0;
  const bool ordinary_counts =
      ordinary_counters.prompt_graph_launches == 0 &&
      ordinary_counters.fused_residual_norm_kernel_launches == 127 &&
      ordinary_counters.residual_add_kernel_launches == 1;

  status = graph_session.reset();
  if (status.is_ok()) status = ordinary_session.reset();
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> fallback_graph_logits(qw38::internal::kVocabularySize);
  std::vector<float> fallback_ordinary_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> fallback_graph_hidden{};
  std::array<float, qw38::internal::kResidualWidth> fallback_ordinary_hidden{};
  status = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), kFallbackRows, &graph_session, &graph_workspace,
      fallback_graph_logits.data(), fallback_graph_logits.size(),
      fallback_graph_hidden.data(), fallback_graph_hidden.size(), nullptr,
      qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr, &graphs);
  if (status.is_ok()) {
    status = qw38::cuda::execute_prompt_chunk(
        model, tokens.data(), kFallbackRows, &ordinary_session,
        &ordinary_workspace, fallback_ordinary_logits.data(),
        fallback_ordinary_logits.size(), fallback_ordinary_hidden.data(),
        fallback_ordinary_hidden.size(), nullptr,
        qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr, nullptr);
  }
  if (!status.is_ok()) return fail_status(status);
  bool fallback_state = false;
  status = graph_session.state_equals(ordinary_session, &fallback_state);
  if (!status.is_ok()) return fail_status(status);
  const bool fallback_ok =
      fallback_state &&
      outputs_equal(fallback_graph_logits, fallback_ordinary_logits,
                    fallback_graph_hidden, fallback_ordinary_hidden) &&
      graph_session.frontier() == kFallbackRows;

  std::vector<float> cancel_logits(qw38::internal::kVocabularySize, 17.0F);
  std::array<float, qw38::internal::kResidualWidth> cancel_hidden{};
  cancel_hidden.fill(23.0F);
  const std::vector<float> cancel_logits_before = cancel_logits;
  const auto cancel_hidden_before = cancel_hidden;
  PollContext cancellation{0, 8};
  const qw38::cuda::EvalControl cancel_control{poll, &cancellation};
  status = graph_session.reset();
  qw38::cuda::SchedulerSession empty_session;
  if (status.is_ok()) status = empty_session.create(kWideRows);
  if (!status.is_ok()) return fail_status(status);
  qw38::cuda::PromptPipelineCounters cancel_counters;
  const qw38::Status cancel_status = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), kWideRows, &graph_session, &graph_workspace,
      cancel_logits.data(), cancel_logits.size(), cancel_hidden.data(),
      cancel_hidden.size(), &cancel_control,
      qw38::cuda::PromptPipelinePath::kFusedOverlapped, &cancel_counters,
      &graphs);
  bool cancel_state_equal = false;
  status = graph_session.state_equals(empty_session, &cancel_state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool cancel_ok =
      cancel_status.code() == qw38::StatusCode::kCancelled &&
      graph_session.frontier() == 0 && cancel_state_equal &&
      outputs_equal(cancel_logits, cancel_logits_before, cancel_hidden,
                    cancel_hidden_before) &&
      cancel_counters.layer_polls == 8 &&
      cancel_counters.scatter_kernel_launches == 0 &&
      cancel_counters.prompt_graph_launches == 8;

  const bool passed = graphs_4096_ok && graphs_65_ok && launch_ok && mismatch_ok &&
                      unfused_ok && equality_ok && fallback_ok && cancel_ok &&
                      graph_counts && ordinary_counts;

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
      "QW38_PROMPT_GRAPH_RESULT={\"schema_version\":1,\"task\":\"OPT-012\","
      "\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\","
      "\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\","
      "\"semantic\":{"
      "\"capacity_4096_owns_128_graphs\":%s,"
      "\"capacity_65_skips_prompt_graphs\":%s,"
      "\"prompt_ffn_kernel_nodes\":%s,"
      "\"prompt_ffn_rowwise_grid\":%s,"
      "\"mismatch_fail_closed\":%s,"
      "\"unfused_fail_closed\":%s,"
      "\"graph_ordinary_exact_4096\":%s,"
      "\"fallback_exact_64\":%s,"
      "\"cancellation_poll_8_publishes_nothing\":%s,"
      "\"graph_counters_match\":%s,"
      "\"ordinary_counters_match\":%s},"
      "\"graphs\":{"
      "\"capacity_4096\":{\"decode_graph_count\":%zu,\"prompt_graph_count\":%zu,"
      "\"prompt_graph_rows\":%zu,\"graph_count\":%zu,\"allocated_bytes\":%zu},"
      "\"capacity_65\":{\"decode_graph_count\":64,\"prompt_graph_count\":0,"
      "\"prompt_graph_rows\":0,\"graph_count\":64}},"
      "\"launch\":{\"kernel_nodes\":%d,\"has_rowwise_residual_add_norm\":%s,"
      "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u]},"
      "\"fail_closed\":{\"mismatch\":%s,\"unfused\":%s,\"mismatch_frontier\":%zu},"
      "\"equality\":{\"state_equals\":%s,\"hidden_memcmp\":%s,\"logits_memcmp\":%s,"
      "\"frontier\":%zu,\"graph_ms\":%.9g,\"ordinary_ms\":%.9g},"
      "\"fallback\":{\"state_hidden_logits_exact\":%s,\"rows\":%zu},"
      "\"cancellation\":{\"polls\":%u,\"frontier\":%zu,"
      "\"scatter_kernel_launches\":%u,\"prompt_graph_launches\":%u,"
      "\"outputs_unchanged\":%s,\"state_equals_empty\":%s},"
      "\"counters\":{"
      "\"graph\":{\"prompt_graph_launches\":%u,\"embedding_kernel_launches\":%u,"
      "\"scatter_kernel_launches\":%u,\"blocking_d2h_copies\":%u,"
      "\"async_d2h_copies\":%u,\"device_synchronizes\":%u},"
      "\"ordinary\":{\"prompt_graph_launches\":%u,"
      "\"fused_residual_norm_kernel_launches\":%u,"
      "\"residual_add_kernel_launches\":%u}},"
      "\"proof_limit\":\"component-only prompt FFN subgraph evidence; not a "
      "whole-chunk graph; Nsight Systems is not claimed; end-to-end "
      "prefill/decode speedup and 128K quality are not claimed\"}\n",
      prop.name, prop.major, prop.minor, driver / 1000, (driver % 1000) / 10,
      runtime / 1000, (runtime % 1000) / 10, utc,
      json_bool(graphs_4096_ok).c_str(), json_bool(graphs_65_ok).c_str(),
      json_bool(prompt_ffn.kernel_nodes >= 6).c_str(),
      json_bool(prompt_ffn.has_rowwise).c_str(), json_bool(mismatch_ok).c_str(),
      json_bool(unfused_ok).c_str(), json_bool(equality_ok).c_str(),
      json_bool(fallback_ok).c_str(), json_bool(cancel_ok).c_str(),
      json_bool(graph_counts).c_str(), json_bool(ordinary_counts).c_str(),
      graphs.decode_graph_count(), graphs.prompt_graph_count(),
      graphs.prompt_graph_rows(), graphs.graph_count(), graphs.allocated_bytes(),
      prompt_ffn.kernel_nodes, json_bool(prompt_ffn.has_rowwise).c_str(),
      prompt_ffn.grid.x, prompt_ffn.grid.y, prompt_ffn.grid.z,
      prompt_ffn.block.x, prompt_ffn.block.y, prompt_ffn.block.z,
      json_bool(mismatch_ok).c_str(), json_bool(unfused_ok).c_str(),
      mismatch_session.frontier(), json_bool(equality_state).c_str(),
      json_bool(hidden_exact).c_str(), json_bool(logits_exact).c_str(),
      equality_frontier,
      graph_ms, ordinary_ms, json_bool(fallback_ok).c_str(), kFallbackRows,
      cancel_counters.layer_polls, graph_session.frontier(),
      cancel_counters.scatter_kernel_launches,
      cancel_counters.prompt_graph_launches,
      json_bool(outputs_equal(cancel_logits, cancel_logits_before, cancel_hidden,
                              cancel_hidden_before))
          .c_str(),
      json_bool(cancel_state_equal).c_str(), graph_counters.prompt_graph_launches,
      graph_counters.embedding_kernel_launches,
      graph_counters.scatter_kernel_launches, graph_counters.blocking_d2h_copies,
      graph_counters.async_d2h_copies, graph_counters.device_synchronizes,
      ordinary_counters.prompt_graph_launches,
      ordinary_counters.fused_residual_norm_kernel_launches,
      ordinary_counters.residual_add_kernel_launches);
  return passed ? 0 : 1;
}
