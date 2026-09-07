#include "full_scheduler.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "attention_decode.h"
#include "model.h"
#include "quant_mmv.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr int kThreads = 256;
constexpr std::size_t kWarmups = 3;
constexpr std::size_t kSamples = 30;
constexpr std::size_t kTimingChunks = kWarmups + kSamples;
constexpr std::size_t kChunkRows = 64;
constexpr std::size_t kWideRows = 4096;
constexpr std::size_t kAttentionLayers = 16;

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
};

struct KernelGraph final {
  bool ok = false;
  int kernel_nodes = 0;
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
    return {qw38::StatusCode::kCancelled, "prompt pipeline test cancellation"};
  }
  return qw38::Status::ok();
}

bool outputs_equal(const std::vector<float>& left_logits,
                   const std::vector<float>& right_logits,
                   const std::array<float, qw38::internal::kResidualWidth>& left_hidden,
                   const std::array<float, qw38::internal::kResidualWidth>& right_hidden) {
  return std::memcmp(left_logits.data(), right_logits.data(),
                     left_logits.size() * sizeof(float)) == 0 &&
         std::memcmp(left_hidden.data(), right_hidden.data(),
                     left_hidden.size() * sizeof(float)) == 0;
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

std::string json_samples(const std::vector<float>& samples) {
  std::string out = "[";
  char buf[64];
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) out += ",";
    std::snprintf(buf, sizeof(buf), "%.9g", samples[index]);
    out += buf;
  }
  out += "]";
  return out;
}

template <typename Fn>
KernelGraph capture(Fn launch) {
  KernelGraph info;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaError_t error =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
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
    info.grid = params.gridDim;
    info.block = params.blockDim;
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  info.ok = error == cudaSuccess;
  return info;
}

void fill_tokens(std::size_t* tokens, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qw38-cuda-prompt-pipeline-test MODEL\n");
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

  std::uint8_t* dummy_weights = nullptr;
  std::size_t* dummy_token_ids = nullptr;
  float* dummy_residual = nullptr;
  float* dummy_correction = nullptr;
  float* dummy_scale = nullptr;
  float* dummy_output = nullptr;
  __nv_bfloat16* dummy_normalized = nullptr;
  __nv_bfloat16* dummy_ck = nullptr;
  __nv_bfloat16* dummy_cv = nullptr;
  __nv_bfloat16* dummy_tk = nullptr;
  __nv_bfloat16* dummy_tv = nullptr;
  cudaError_t alloc = cudaMalloc(&dummy_weights, 256);
  if (alloc == cudaSuccess) {
    alloc = cudaMalloc(&dummy_token_ids, kWideRows * sizeof(std::size_t));
  }
  if (alloc == cudaSuccess) {
    alloc = cudaMalloc(&dummy_residual, kWideRows * qw38::internal::kResidualWidth *
                                            sizeof(float));
  }
  if (alloc == cudaSuccess) {
    alloc = cudaMalloc(&dummy_correction,
                       kChunkRows * qw38::internal::kResidualWidth * sizeof(float));
  }
  if (alloc == cudaSuccess) {
    alloc = cudaMalloc(&dummy_scale, qw38::internal::kResidualWidth * sizeof(float));
  }
  if (alloc == cudaSuccess) {
    alloc = cudaMalloc(&dummy_output,
                       kChunkRows * qw38::internal::kResidualWidth * sizeof(float));
  }
  if (alloc == cudaSuccess) {
    alloc = cudaMalloc(&dummy_normalized, kChunkRows *
                                              qw38::internal::kResidualWidth *
                                              sizeof(__nv_bfloat16));
  }
  if (alloc == cudaSuccess) alloc = cudaMalloc(&dummy_ck, 256);
  if (alloc == cudaSuccess) alloc = cudaMalloc(&dummy_cv, 256);
  if (alloc == cudaSuccess) alloc = cudaMalloc(&dummy_tk, 256);
  if (alloc == cudaSuccess) alloc = cudaMalloc(&dummy_tv, 256);
  if (alloc != cudaSuccess) {
    std::fprintf(stderr, "cannot allocate prompt pipeline capture buffers\n");
    return 1;
  }

  const KernelGraph fused_embed_64 = capture([&](cudaStream_t stream) {
    return qw38::cuda::launch_quant_rows_decode_widen(
        qw38::cuda::QuantKind::kQ6K, dummy_weights, kWideRows,
        qw38::internal::kResidualWidth, dummy_token_ids, kChunkRows,
        dummy_residual, stream);
  });
  const KernelGraph fused_embed_4096 = capture([&](cudaStream_t stream) {
    return qw38::cuda::launch_quant_rows_decode_widen(
        qw38::cuda::QuantKind::kQ6K, dummy_weights, kWideRows,
        qw38::internal::kResidualWidth, dummy_token_ids, kWideRows,
        dummy_residual, stream);
  });
  const KernelGraph unfused_embed_64 = capture([&](cudaStream_t stream) {
    cudaError_t error = cudaSuccess;
    for (std::size_t row = 0; error == cudaSuccess && row < kChunkRows; ++row) {
      error = qw38::cuda::launch_quant_row_decode(
          qw38::cuda::QuantKind::kQ6K, dummy_weights, kWideRows,
          qw38::internal::kResidualWidth, row % kWideRows,
          dummy_normalized + (row % kChunkRows) * qw38::internal::kResidualWidth,
          stream);
    }
    return error;
  });
  const KernelGraph fused_residual = capture([&](cudaStream_t stream) {
    return qw38::cuda::launch_residual_add_norm_rows_fp32_to_bf16(
        dummy_residual, dummy_correction, dummy_scale,
        qw38::internal::kResidualWidth, kChunkRows, dummy_output,
        dummy_normalized, stream);
  });
  const KernelGraph unfused_residual = capture([&](cudaStream_t stream) {
    cudaError_t error = qw38::cuda::launch_residual_add_fp32(
        dummy_residual, dummy_correction,
        kChunkRows * qw38::internal::kResidualWidth, dummy_output, stream);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_rms_norm_rows_fp32_to_bf16(
          dummy_output, dummy_scale, qw38::internal::kResidualWidth, kChunkRows,
          dummy_normalized, stream);
    }
    return error;
  });
  const qw38::cuda::AttentionConfig scatter_config{24, 4, 256, 64, kWideRows};
  const KernelGraph fused_scatter_64 = capture([&](cudaStream_t stream) {
    return qw38::cuda::launch_attention_scatter_layers(
        scatter_config, 0, kChunkRows, kAttentionLayers, dummy_tk, dummy_tv,
        kChunkRows * qw38::internal::kAttentionKvWidth, dummy_ck, dummy_cv,
        kWideRows * qw38::internal::kAttentionKvWidth, stream);
  });
  const KernelGraph fused_scatter_4096 = capture([&](cudaStream_t stream) {
    return qw38::cuda::launch_attention_scatter_layers(
        scatter_config, 0, kWideRows, kAttentionLayers, dummy_tk, dummy_tv,
        kWideRows * qw38::internal::kAttentionKvWidth, dummy_ck, dummy_cv,
        kWideRows * qw38::internal::kAttentionKvWidth, stream);
  });
  const KernelGraph unfused_scatter_64 = capture([&](cudaStream_t stream) {
    cudaError_t error = cudaSuccess;
    for (std::size_t layer = 0; error == cudaSuccess && layer < kAttentionLayers;
         ++layer) {
      error = qw38::cuda::launch_attention_scatter_chunk(
          scatter_config, 0, kChunkRows, {dummy_tk, dummy_tv}, {dummy_ck, dummy_cv},
          stream);
    }
    return error;
  });

  cudaFree(dummy_tv);
  cudaFree(dummy_tk);
  cudaFree(dummy_cv);
  cudaFree(dummy_ck);
  cudaFree(dummy_normalized);
  cudaFree(dummy_output);
  cudaFree(dummy_scale);
  cudaFree(dummy_correction);
  cudaFree(dummy_residual);
  cudaFree(dummy_token_ids);
  cudaFree(dummy_weights);

  const bool launch_ok =
      fused_embed_64.ok && fused_embed_4096.ok && unfused_embed_64.ok &&
      fused_residual.ok && unfused_residual.ok && fused_scatter_64.ok &&
      fused_scatter_4096.ok && unfused_scatter_64.ok &&
      fused_embed_64.kernel_nodes == 1 && fused_embed_4096.kernel_nodes == 1 &&
      fused_embed_64.grid.y == kChunkRows &&
      fused_embed_4096.grid.y == kWideRows && fused_embed_64.block.x == kThreads &&
      fused_residual.kernel_nodes == 1 && unfused_residual.kernel_nodes == 2 &&
      fused_scatter_64.kernel_nodes == 1 &&
      fused_scatter_4096.kernel_nodes == 1 && fused_scatter_64.grid.y == 16 &&
      fused_scatter_4096.grid.y == 16 && fused_scatter_64.grid.x >= 2 &&
      fused_scatter_4096.grid.x >= 16 && unfused_scatter_64.kernel_nodes == 16 &&
      unfused_embed_64.kernel_nodes == static_cast<int>(kChunkRows);

  std::array<std::size_t, kWideRows + 1> tokens{};
  fill_tokens(tokens.data(), tokens.size());
  const std::size_t equality_rows[] = {2, 3, 64, 65};
  bool equality_ok = true;
  bool equality_flags[4] = {false, false, false, false};
  qw38::cuda::PromptPipelineCounters fused_counters;
  qw38::cuda::PromptPipelineCounters unfused_counters;
  for (std::size_t case_index = 0; case_index < 4; ++case_index) {
    const std::size_t rows = equality_rows[case_index];
    const std::size_t capacity = rows;
    qw38::cuda::SchedulerSession fused_session;
    qw38::cuda::SchedulerSession unfused_session;
    qw38::cuda::SchedulerWorkspace fused_workspace;
    qw38::cuda::SchedulerWorkspace unfused_workspace;
    if (status.is_ok()) status = fused_session.create(capacity);
    if (status.is_ok()) status = unfused_session.create(capacity);
    if (status.is_ok()) status = fused_workspace.create(capacity);
    if (status.is_ok()) status = unfused_workspace.create(capacity);
    if (!status.is_ok()) return fail_status(status);
    std::vector<float> fused_logits(qw38::internal::kVocabularySize);
    std::vector<float> unfused_logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> fused_hidden{};
    std::array<float, qw38::internal::kResidualWidth> unfused_hidden{};
    qw38::cuda::PromptPipelineCounters* fused_counter_ptr =
        rows == kChunkRows ? &fused_counters : nullptr;
    qw38::cuda::PromptPipelineCounters* unfused_counter_ptr =
        rows == kChunkRows ? &unfused_counters : nullptr;
    status = qw38::cuda::execute_prompt_chunk(
        model, tokens.data(), rows, &fused_session, &fused_workspace,
        fused_logits.data(), fused_logits.size(), fused_hidden.data(),
        fused_hidden.size(), nullptr, qw38::cuda::PromptPipelinePath::kFusedOverlapped,
        fused_counter_ptr);
    if (status.is_ok()) {
      status = qw38::cuda::execute_prompt_chunk(
          model, tokens.data(), rows, &unfused_session, &unfused_workspace,
          unfused_logits.data(), unfused_logits.size(), unfused_hidden.data(),
          unfused_hidden.size(), nullptr,
          qw38::cuda::PromptPipelinePath::kUnfusedSerial, unfused_counter_ptr);
    }
    if (!status.is_ok()) return fail_status(status);
    bool state_equal = false;
    status = fused_session.state_equals(unfused_session, &state_equal);
    if (!status.is_ok()) return fail_status(status);
    const bool hidden_exact =
        std::memcmp(fused_hidden.data(), unfused_hidden.data(),
                    fused_hidden.size() * sizeof(float)) == 0;
    const bool logits_exact =
        std::memcmp(fused_logits.data(), unfused_logits.data(),
                    fused_logits.size() * sizeof(float)) == 0;
    equality_flags[case_index] = state_equal && hidden_exact && logits_exact &&
                                 fused_session.frontier() == rows;
    equality_ok = equality_ok && equality_flags[case_index];
  }

  const bool fused_counts =
      fused_counters.embedding_kernel_launches == 1 &&
      fused_counters.widen_kernel_launches == 0 &&
      fused_counters.scatter_kernel_launches == 1 &&
      fused_counters.blocking_d2h_copies == 0 &&
      fused_counters.async_d2h_copies == 2 &&
      fused_counters.device_synchronizes == 0 &&
      fused_counters.fused_residual_norm_kernel_launches == 127 &&
      fused_counters.residual_add_kernel_launches == 1;
  const bool unfused_counts =
      unfused_counters.embedding_kernel_launches == 64 &&
      unfused_counters.widen_kernel_launches == 1 &&
      unfused_counters.scatter_kernel_launches == 16 &&
      unfused_counters.blocking_d2h_copies == 2 &&
      unfused_counters.device_synchronizes == 1;

  std::vector<float> cancel_logits(qw38::internal::kVocabularySize, 17.0F);
  std::array<float, qw38::internal::kResidualWidth> cancel_hidden{};
  cancel_hidden.fill(23.0F);
  const std::vector<float> cancel_logits_before = cancel_logits;
  const auto cancel_hidden_before = cancel_hidden;
  PollContext cancellation{0, 8};
  const qw38::cuda::EvalControl cancel_control{poll, &cancellation};
  qw38::cuda::SchedulerSession cancel_session;
  qw38::cuda::SchedulerSession empty_session;
  qw38::cuda::SchedulerWorkspace cancel_workspace;
  status = cancel_session.create(kChunkRows);
  if (status.is_ok()) status = empty_session.create(kChunkRows);
  if (status.is_ok()) status = cancel_workspace.create(kChunkRows);
  if (!status.is_ok()) return fail_status(status);
  qw38::cuda::PromptPipelineCounters cancel_counters;
  const qw38::Status cancel_status = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), kChunkRows, &cancel_session, &cancel_workspace,
      cancel_logits.data(), cancel_logits.size(), cancel_hidden.data(),
      cancel_hidden.size(), &cancel_control,
      qw38::cuda::PromptPipelinePath::kFusedOverlapped, &cancel_counters);
  bool cancel_state_equal = false;
  status = cancel_session.state_equals(empty_session, &cancel_state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool cancel_ok =
      cancel_status.code() == qw38::StatusCode::kCancelled &&
      cancel_session.frontier() == 0 && cancel_state_equal &&
      outputs_equal(cancel_logits, cancel_logits_before, cancel_hidden,
                    cancel_hidden_before) &&
      cancel_counters.layer_polls == 8 &&
      cancel_counters.scatter_kernel_launches == 0;

  qw38::cuda::SchedulerSession invalid_session;
  qw38::cuda::SchedulerWorkspace invalid_workspace;
  status = invalid_session.create(kChunkRows);
  if (status.is_ok()) status = invalid_workspace.create(kChunkRows);
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> invalid_logits(qw38::internal::kVocabularySize, 5.0F);
  std::array<float, qw38::internal::kResidualWidth> invalid_hidden{};
  invalid_hidden.fill(9.0F);
  const std::vector<float> invalid_logits_before = invalid_logits;
  const auto invalid_hidden_before = invalid_hidden;
  qw38::cuda::ResidentModel empty_model;
  const qw38::Status null_model = qw38::cuda::execute_prompt_chunk(
      empty_model, tokens.data(), 2, &invalid_session, &invalid_workspace,
      invalid_logits.data(), invalid_logits.size(), invalid_hidden.data(),
      invalid_hidden.size());
  const qw38::Status null_tokens = qw38::cuda::execute_prompt_chunk(
      model, nullptr, 2, &invalid_session, &invalid_workspace,
      invalid_logits.data(), invalid_logits.size(), invalid_hidden.data(),
      invalid_hidden.size());
  const qw38::Status one_token = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), 1, &invalid_session, &invalid_workspace,
      invalid_logits.data(), invalid_logits.size(), invalid_hidden.data(),
      invalid_hidden.size());
  const qw38::Status too_many = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), invalid_workspace.prompt_chunk_rows_ + 1,
      &invalid_session, &invalid_workspace, invalid_logits.data(),
      invalid_logits.size(), invalid_hidden.data(), invalid_hidden.size());
  const qw38::Status bad_path = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), 2, &invalid_session, &invalid_workspace,
      invalid_logits.data(), invalid_logits.size(), invalid_hidden.data(),
      invalid_hidden.size(), nullptr,
      static_cast<qw38::cuda::PromptPipelinePath>(255), nullptr);
  const bool invalid_ok =
      null_model.code() == qw38::StatusCode::kInvalidArgument &&
      null_tokens.code() == qw38::StatusCode::kInvalidArgument &&
      one_token.code() == qw38::StatusCode::kInvalidArgument &&
      too_many.code() == qw38::StatusCode::kInvalidArgument &&
      bad_path.code() == qw38::StatusCode::kInvalidArgument &&
      invalid_session.frontier() == 0 &&
      outputs_equal(invalid_logits, invalid_logits_before, invalid_hidden,
                    invalid_hidden_before);

  constexpr std::size_t kTimingCapacity = kTimingChunks * kChunkRows;
  qw38::cuda::SchedulerSession fused_timing_session;
  qw38::cuda::SchedulerSession unfused_timing_session;
  qw38::cuda::SchedulerWorkspace fused_timing_workspace;
  qw38::cuda::SchedulerWorkspace unfused_timing_workspace;
  status = fused_timing_session.create(kTimingCapacity);
  if (status.is_ok()) status = unfused_timing_session.create(kTimingCapacity);
  if (status.is_ok()) status = fused_timing_workspace.create(kTimingCapacity);
  if (status.is_ok()) status = unfused_timing_workspace.create(kTimingCapacity);
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> fused_timing_logits(qw38::internal::kVocabularySize);
  std::vector<float> unfused_timing_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> fused_timing_hidden{};
  std::array<float, qw38::internal::kResidualWidth> unfused_timing_hidden{};
  std::vector<float> fused_ms;
  std::vector<float> unfused_ms;
  fused_ms.reserve(kSamples);
  unfused_ms.reserve(kSamples);
  cudaEvent_t fused_start = nullptr;
  cudaEvent_t fused_stop = nullptr;
  cudaEvent_t unfused_start = nullptr;
  cudaEvent_t unfused_stop = nullptr;
  cudaError_t event_error = cudaEventCreate(&fused_start);
  if (event_error == cudaSuccess) event_error = cudaEventCreate(&fused_stop);
  if (event_error == cudaSuccess) event_error = cudaEventCreate(&unfused_start);
  if (event_error == cudaSuccess) event_error = cudaEventCreate(&unfused_stop);
  if (event_error != cudaSuccess) {
    std::fprintf(stderr, "cannot create prompt pipeline timing events\n");
    return 1;
  }
  auto run_fused = [&]() -> qw38::Status {
    event_error =
        cudaEventRecord(fused_start, fused_timing_workspace.prompt_compute_stream_);
    if (event_error != cudaSuccess) {
      return {qw38::StatusCode::kInternal, "fused timing start failed"};
    }
    qw38::Status chunk = qw38::cuda::execute_prompt_chunk(
        model, tokens.data(), kChunkRows,
        &fused_timing_session, &fused_timing_workspace,
        fused_timing_logits.data(), fused_timing_logits.size(),
        fused_timing_hidden.data(), fused_timing_hidden.size(), nullptr,
        qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr);
    if (!chunk.is_ok()) return chunk;
    event_error =
        cudaEventRecord(fused_stop, fused_timing_workspace.prompt_compute_stream_);
    if (event_error == cudaSuccess) event_error = cudaEventSynchronize(fused_stop);
    return event_error == cudaSuccess
               ? qw38::Status::ok()
               : qw38::Status{qw38::StatusCode::kInternal, "fused timing stop failed"};
  };
  auto run_unfused = [&]() -> qw38::Status {
    event_error = cudaEventRecord(unfused_start);
    if (event_error != cudaSuccess) {
      return {qw38::StatusCode::kInternal, "unfused timing start failed"};
    }
    qw38::Status chunk = qw38::cuda::execute_prompt_chunk(
        model, tokens.data(), kChunkRows,
        &unfused_timing_session, &unfused_timing_workspace,
        unfused_timing_logits.data(), unfused_timing_logits.size(),
        unfused_timing_hidden.data(), unfused_timing_hidden.size(), nullptr,
        qw38::cuda::PromptPipelinePath::kUnfusedSerial, nullptr);
    if (!chunk.is_ok()) return chunk;
    event_error = cudaEventRecord(unfused_stop);
    if (event_error == cudaSuccess) event_error = cudaEventSynchronize(unfused_stop);
    return event_error == cudaSuccess
               ? qw38::Status::ok()
               : qw38::Status{qw38::StatusCode::kInternal,
                              "unfused timing stop failed"};
  };
  for (std::size_t sample = 0; sample < kTimingChunks; ++sample) {
    float fused_elapsed = 0.0F;
    float unfused_elapsed = 0.0F;
    if (sample % 2 == 0) {
      status = run_fused();
      if (status.is_ok()) {
        event_error = cudaEventElapsedTime(&fused_elapsed, fused_start, fused_stop);
      }
      if (status.is_ok() && event_error == cudaSuccess) status = run_unfused();
      if (status.is_ok()) {
        event_error =
            cudaEventElapsedTime(&unfused_elapsed, unfused_start, unfused_stop);
      }
    } else {
      status = run_unfused();
      if (status.is_ok()) {
        event_error =
            cudaEventElapsedTime(&unfused_elapsed, unfused_start, unfused_stop);
      }
      if (status.is_ok() && event_error == cudaSuccess) status = run_fused();
      if (status.is_ok()) {
        event_error = cudaEventElapsedTime(&fused_elapsed, fused_start, fused_stop);
      }
    }
    if (!status.is_ok()) return fail_status(status);
    if (event_error != cudaSuccess) {
      std::fprintf(stderr, "prompt pipeline timing elapsed failed\n");
      return 1;
    }
    if (sample >= kWarmups) {
      fused_ms.push_back(fused_elapsed);
      unfused_ms.push_back(unfused_elapsed);
    }
  }
  cudaEventDestroy(unfused_stop);
  cudaEventDestroy(unfused_start);
  cudaEventDestroy(fused_stop);
  cudaEventDestroy(fused_start);
  float fused_sum = 0.0F;
  float unfused_sum = 0.0F;
  for (float value : fused_ms) fused_sum += value;
  for (float value : unfused_ms) unfused_sum += value;
  const float fused_mean = fused_sum / static_cast<float>(kSamples);
  const float unfused_mean = unfused_sum / static_cast<float>(kSamples);
  const bool timing_ok = fused_ms.size() == kSamples &&
                         unfused_ms.size() == kSamples && fused_mean < unfused_mean;

  const bool passed = launch_ok && equality_ok && fused_counts && unfused_counts &&
                      cancel_ok && invalid_ok && timing_ok;

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
  const std::string fused_samples = json_samples(fused_ms);
  const std::string unfused_samples = json_samples(unfused_ms);
  std::printf(
      "QW38_PROMPT_PIPELINE_RESULT={\"schema_version\":1,\"task\":\"OPT-011\","
      "\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\","
      "\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\","
      "\"threads\":256,\"semantic\":{"
      "\"fused_unfused_exact_2\":%s,\"fused_unfused_exact_3\":%s,"
      "\"fused_unfused_exact_64\":%s,\"fused_unfused_exact_65\":%s,"
      "\"cancellation_poll_8_publishes_nothing\":%s,"
      "\"cancellation_no_scatter\":%s,\"invalid_input_rejected\":%s,"
      "\"fused_embedding_one_kernel\":%s,\"fused_scatter_one_kernel\":%s,"
      "\"fused_scatter_multi_block\":%s,\"fused_mean_below_unfused\":%s,"
      "\"fused_counters_match\":%s,\"unfused_counters_match\":%s},"
      "\"equality\":{\"row_counts\":[2,3,64,65],\"state_hidden_logits_exact\":[%s,"
      "%s,%s,%s]},\"cancellation\":{\"polls\":%u,\"frontier\":%zu,"
      "\"scatter_kernel_launches\":%u,\"outputs_unchanged\":%s,"
      "\"state_equals_empty\":%s},\"launch\":{"
      "\"fused_embedding_64\":{\"kernel_nodes\":%d,\"grid\":[%u,%u,%u],"
      "\"block\":[%u,%u,%u]},"
      "\"fused_embedding_4096\":{\"kernel_nodes\":%d,\"grid\":[%u,%u,%u],"
      "\"block\":[%u,%u,%u]},"
      "\"unfused_embedding_64\":{\"kernel_nodes\":%d},"
      "\"fused_residual_norm\":{\"kernel_nodes\":%d,\"grid\":[%u,%u,%u],"
      "\"block\":[%u,%u,%u]},"
      "\"unfused_residual_norm\":{\"kernel_nodes\":%d},"
      "\"fused_scatter_64\":{\"kernel_nodes\":%d,\"grid\":[%u,%u,%u],"
      "\"block\":[%u,%u,%u]},"
      "\"fused_scatter_4096\":{\"kernel_nodes\":%d,\"grid\":[%u,%u,%u],"
      "\"block\":[%u,%u,%u]},"
      "\"unfused_scatter_64\":{\"kernel_nodes\":%d}},"
      "\"counters\":{\"fused\":{\"embedding_kernel_launches\":%u,"
      "\"widen_kernel_launches\":%u,\"scatter_kernel_launches\":%u,"
      "\"blocking_d2h_copies\":%u,\"async_d2h_copies\":%u,"
      "\"device_synchronizes\":%u,\"fused_residual_norm_kernel_launches\":%u,"
      "\"residual_add_kernel_launches\":%u},"
      "\"unfused\":{\"embedding_kernel_launches\":%u,\"widen_kernel_launches\":%u,"
      "\"scatter_kernel_launches\":%u,\"blocking_d2h_copies\":%u,"
      "\"async_d2h_copies\":%u,\"device_synchronizes\":%u}},"
      "\"timing\":{\"warmups\":%zu,\"samples\":%zu,\"fused_mean_ms\":%.9g,"
      "\"unfused_mean_ms\":%.9g,\"fused_ms\":%s,\"unfused_ms\":%s},"
      "\"proof_limit\":\"component-only orchestration evidence; no Nsight "
      "Systems overlap screenshot; no end-to-end prefill/decode speedup\"}\n",
      prop.name, prop.major, prop.minor, driver / 1000, (driver % 1000) / 10,
      runtime / 1000, (runtime % 1000) / 10, utc,
      json_bool(equality_flags[0]).c_str(), json_bool(equality_flags[1]).c_str(),
      json_bool(equality_flags[2]).c_str(), json_bool(equality_flags[3]).c_str(),
      json_bool(cancel_ok).c_str(),
      json_bool(cancel_counters.scatter_kernel_launches == 0).c_str(),
      json_bool(invalid_ok).c_str(),
      json_bool(fused_embed_64.kernel_nodes == 1 &&
                fused_embed_4096.kernel_nodes == 1)
          .c_str(),
      json_bool(fused_scatter_64.kernel_nodes == 1 &&
                fused_scatter_4096.kernel_nodes == 1)
          .c_str(),
      json_bool(fused_scatter_64.grid.x >= 2 && fused_scatter_4096.grid.x >= 16)
          .c_str(),
      json_bool(timing_ok).c_str(), json_bool(fused_counts).c_str(),
      json_bool(unfused_counts).c_str(), json_bool(equality_flags[0]).c_str(),
      json_bool(equality_flags[1]).c_str(), json_bool(equality_flags[2]).c_str(),
      json_bool(equality_flags[3]).c_str(), cancel_counters.layer_polls,
      cancel_session.frontier(), cancel_counters.scatter_kernel_launches,
      json_bool(outputs_equal(cancel_logits, cancel_logits_before, cancel_hidden,
                              cancel_hidden_before))
          .c_str(),
      json_bool(cancel_state_equal).c_str(), fused_embed_64.kernel_nodes,
      fused_embed_64.grid.x, fused_embed_64.grid.y, fused_embed_64.grid.z,
      fused_embed_64.block.x, fused_embed_64.block.y, fused_embed_64.block.z,
      fused_embed_4096.kernel_nodes, fused_embed_4096.grid.x,
      fused_embed_4096.grid.y, fused_embed_4096.grid.z, fused_embed_4096.block.x,
      fused_embed_4096.block.y, fused_embed_4096.block.z,
      unfused_embed_64.kernel_nodes, fused_residual.kernel_nodes,
      fused_residual.grid.x, fused_residual.grid.y, fused_residual.grid.z,
      fused_residual.block.x, fused_residual.block.y, fused_residual.block.z,
      unfused_residual.kernel_nodes, fused_scatter_64.kernel_nodes,
      fused_scatter_64.grid.x, fused_scatter_64.grid.y, fused_scatter_64.grid.z,
      fused_scatter_64.block.x, fused_scatter_64.block.y, fused_scatter_64.block.z,
      fused_scatter_4096.kernel_nodes, fused_scatter_4096.grid.x,
      fused_scatter_4096.grid.y, fused_scatter_4096.grid.z,
      fused_scatter_4096.block.x, fused_scatter_4096.block.y,
      fused_scatter_4096.block.z, unfused_scatter_64.kernel_nodes,
      fused_counters.embedding_kernel_launches,
      fused_counters.widen_kernel_launches, fused_counters.scatter_kernel_launches,
      fused_counters.blocking_d2h_copies, fused_counters.async_d2h_copies,
      fused_counters.device_synchronizes,
      fused_counters.fused_residual_norm_kernel_launches,
      fused_counters.residual_add_kernel_launches,
      unfused_counters.embedding_kernel_launches,
      unfused_counters.widen_kernel_launches,
      unfused_counters.scatter_kernel_launches,
      unfused_counters.blocking_d2h_copies, unfused_counters.async_d2h_copies,
      unfused_counters.device_synchronizes, kWarmups, kSamples, fused_mean,
      unfused_mean, fused_samples.c_str(), unfused_samples.c_str());
  return passed ? 0 : 1;
}
