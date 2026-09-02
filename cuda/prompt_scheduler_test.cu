#include "full_scheduler.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

qw38::Status cancel(void*) noexcept {
  return {qw38::StatusCode::kCancelled, "prompt test cancellation"};
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qw38-cuda-prompt-scheduler-test MODEL\n");
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

  constexpr std::size_t kCount = qw38::cuda::kPromptChunkRows + 1;
  qw38::cuda::SchedulerSession optimized;
  qw38::cuda::SchedulerSession reference;
  qw38::cuda::SchedulerSession untouched;
  if (status.is_ok()) status = optimized.create(kCount);
  if (status.is_ok()) status = reference.create(kCount);
  if (status.is_ok()) status = untouched.create(kCount);
  qw38::cuda::SchedulerWorkspace optimized_workspace;
  qw38::cuda::SchedulerWorkspace reference_workspace;
  qw38::cuda::SchedulerWorkspace cancelled_workspace;
  if (status.is_ok()) status = optimized_workspace.create(kCount);
  if (status.is_ok()) status = reference_workspace.create(kCount);
  if (status.is_ok()) status = cancelled_workspace.create(kCount);
  if (!status.is_ok()) return fail_status(status);

  std::array<std::size_t, kCount> tokens{};
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
  std::vector<float> optimized_logits(qw38::internal::kVocabularySize);
  std::vector<float> reference_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> optimized_hidden{};
  std::array<float, qw38::internal::kResidualWidth> reference_hidden{};
  qw38::cuda::SyncResult sync_result;

  const auto optimized_start = std::chrono::steady_clock::now();
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), tokens.size(), &optimized, &optimized_workspace,
      optimized_logits.data(), optimized_logits.size(),
      optimized_hidden.data(), optimized_hidden.size(), &sync_result);
  const double optimized_ms = elapsed_ms(optimized_start);
  if (!status.is_ok()) return fail_status(status);

  float token_ms = 0.0F;
  const auto reference_start = std::chrono::steady_clock::now();
  for (std::size_t token : tokens) {
    status = qw38::cuda::execute_token(
        model, token, &reference, &reference_workspace,
        reference_logits.data(), reference_logits.size(),
        reference_hidden.data(), reference_hidden.size(), &token_ms);
    if (!status.is_ok()) return fail_status(status);
  }
  const double reference_ms = elapsed_ms(reference_start);
  bool state_equal = false;
  status = optimized.state_equals(reference, &state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool outputs_equal =
      std::memcmp(optimized_logits.data(), reference_logits.data(),
                  optimized_logits.size() * sizeof(float)) == 0 &&
      std::memcmp(optimized_hidden.data(), reference_hidden.data(),
                  optimized_hidden.size() * sizeof(float)) == 0;
  const bool faster = optimized_ms < reference_ms;
  std::printf(
      "prompt_chunk=tokens_65 rows_per_chunk=%zu evaluated=%zu frontier=%zu "
      "state_exact=%s outputs_exact=%s optimized_ms=%.6f "
      "tokenwise_ms=%.6f speedup=%.6f passed=%s\n",
      qw38::cuda::kPromptChunkRows, sync_result.evaluated_tokens,
      optimized.frontier(), state_equal ? "true" : "false",
      outputs_equal ? "true" : "false", optimized_ms, reference_ms,
      reference_ms / optimized_ms,
      state_equal && outputs_equal && faster ? "true" : "false");

  qw38::cuda::SyncResult cancelled_result;
  const qw38::cuda::EvalControl control{cancel, nullptr};
  const qw38::Status cancel_status = qw38::cuda::sync_tokens(
      model, tokens.data(), qw38::cuda::kPromptChunkRows, &untouched,
      &cancelled_workspace, optimized_logits.data(), optimized_logits.size(),
      optimized_hidden.data(), optimized_hidden.size(), &cancelled_result,
      &control);
  status = reference.reset();
  bool cancellation_state_equal = false;
  if (status.is_ok()) {
    status = untouched.state_equals(reference, &cancellation_state_equal);
  }
  if (!status.is_ok()) return fail_status(status);
  const bool cancelled =
      cancel_status.code() == qw38::StatusCode::kCancelled &&
      untouched.frontier() == 0 && cancellation_state_equal;
  std::printf(
      "prompt_cancel=before_commit status=%s frontier=%zu passed=%s\n",
      qw38::status_code_name(cancel_status.code()), untouched.frontier(),
      cancelled ? "true" : "false");
  std::printf("prompt_workspace_bytes=%zu\n",
              optimized_workspace.allocated_bytes());
  const bool passed = state_equal && outputs_equal && faster && cancelled;
  std::printf("status=%s\n", passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
