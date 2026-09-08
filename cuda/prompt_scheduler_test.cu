#include "full_scheduler.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
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
    return {qw38::StatusCode::kCancelled, "prompt test cancellation"};
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
  if (status.is_ok()) status = qw38::internal::bind_model_weights(info, mapping, &weights);
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
  if (status.is_ok()) status = optimized_workspace.create(kCount);
  if (status.is_ok()) status = reference_workspace.create(kCount);
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
  status = qw38::cuda::sync_tokens(model, tokens.data(), tokens.size(), &optimized,
      &optimized_workspace, optimized_logits.data(), optimized_logits.size(),
      optimized_hidden.data(), optimized_hidden.size(), &sync_result, nullptr,
      nullptr, qw38::cuda::GdnScanPath::kSequentialWindows);
  if (!status.is_ok()) return fail_status(status);

  for (std::size_t index = 0; index < qw38::cuda::kPromptChunkRows; index += 64) {
    status = qw38::cuda::execute_prompt_chunk(model, tokens.data() + index, 64,
        &reference, &reference_workspace, reference_logits.data(),
        reference_logits.size(), reference_hidden.data(), reference_hidden.size(),
        nullptr, qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr, nullptr,
        qw38::cuda::GdnScanPath::kSequentialWindows);
    if (!status.is_ok()) return fail_status(status);
  }
  float token_ms = 0.0F;
  status = qw38::cuda::execute_token(model, tokens.back(), &reference,
      &reference_workspace, reference_logits.data(), reference_logits.size(),
      reference_hidden.data(), reference_hidden.size(), &token_ms);
  if (!status.is_ok()) return fail_status(status);
  bool state_equal = false;
  status = optimized.state_equals(reference, &state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool outer_outputs_equal = outputs_equal(
      optimized_logits, reference_logits, optimized_hidden, reference_hidden);
  const bool outer_passed = sync_result.evaluated_tokens == kCount &&
                            optimized.frontier() == kCount && state_equal &&
                            outer_outputs_equal;
  std::printf("prompt_chunk=tokens_4097 rows_per_chunk=%zu chunks=4096,1 "
              "evaluated=%zu frontier=%zu state_exact=%s outputs_exact=%s passed=%s\n",
              qw38::cuda::kPromptChunkRows, sync_result.evaluated_tokens,
              optimized.frontier(), state_equal ? "true" : "false",
              outer_outputs_equal ? "true" : "false", outer_passed ? "true" : "false");

  constexpr std::size_t kSmallCapacity = 65;
  qw38::cuda::SchedulerSession small;
  qw38::cuda::SchedulerSession small_reference;
  qw38::cuda::SchedulerWorkspace small_workspace;
  qw38::cuda::SchedulerWorkspace small_reference_workspace;
  if (status.is_ok()) status = small.create(kSmallCapacity);
  if (status.is_ok()) status = small_reference.create(kSmallCapacity);
  if (status.is_ok()) status = small_workspace.create(kSmallCapacity);
  if (status.is_ok()) status = small_reference_workspace.create(kSmallCapacity);
  if (!status.is_ok()) return fail_status(status);
  PollContext small_poll;
  const qw38::cuda::EvalControl small_control{poll, &small_poll};
  qw38::cuda::SyncResult small_result;
  status = qw38::cuda::sync_tokens(model, tokens.data(), kSmallCapacity, &small,
      &small_workspace, optimized_logits.data(), optimized_logits.size(),
      optimized_hidden.data(), optimized_hidden.size(), &small_result,
      &small_control, nullptr, qw38::cuda::GdnScanPath::kSequentialWindows);
  if (!status.is_ok()) return fail_status(status);
  status = qw38::cuda::execute_prompt_chunk(model, tokens.data(), kSmallCapacity,
      &small_reference, &small_reference_workspace, reference_logits.data(),
      reference_logits.size(), reference_hidden.data(), reference_hidden.size(),
      nullptr, qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr, nullptr,
      qw38::cuda::GdnScanPath::kSequentialWindows);
  if (!status.is_ok()) return fail_status(status);
  bool small_state_equal = false;
  status = small.state_equals(small_reference, &small_state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool small_outputs_equal = outputs_equal(
      optimized_logits, reference_logits, optimized_hidden, reference_hidden);
  const bool small_passed = small_poll.calls == 64 &&
                            small_result.evaluated_tokens == kSmallCapacity &&
                            small.frontier() == kSmallCapacity && small_state_equal &&
                            small_outputs_equal;
  std::printf("prompt_capacity_fallback=65 chunks=65 poll_calls=%zu evaluated=%zu "
              "frontier=%zu state_exact=%s outputs_exact=%s passed=%s\n",
              small_poll.calls, small_result.evaluated_tokens, small.frontier(),
              small_state_equal ? "true" : "false",
              small_outputs_equal ? "true" : "false", small_passed ? "true" : "false");

  std::fill(optimized_logits.begin(), optimized_logits.end(), 17.0F);
  optimized_hidden.fill(23.0F);
  const std::vector<float> untouched_logits = optimized_logits;
  const auto untouched_hidden = optimized_hidden;
  const qw38::Status bounds_status = qw38::cuda::execute_prompt_chunk(
      model, tokens.data(), kCount, &untouched, &optimized_workspace,
      optimized_logits.data(), optimized_logits.size(), optimized_hidden.data(),
      optimized_hidden.size());
  const bool bounds_passed = bounds_status.code() == qw38::StatusCode::kInvalidArgument &&
      untouched.frontier() == 0 && outputs_equal(optimized_logits, untouched_logits,
      optimized_hidden, untouched_hidden);
  std::printf("prompt_bounds=4097 status=%s frontier=%zu passed=%s\n",
              qw38::status_code_name(bounds_status.code()), untouched.frontier(),
              bounds_passed ? "true" : "false");

  PollContext cancellation{0, 8};
  const qw38::cuda::EvalControl cancel_control{poll, &cancellation};
  qw38::cuda::SyncResult cancelled_result;
  const qw38::Status cancel_status = qw38::cuda::sync_tokens(
      model, tokens.data(), qw38::cuda::kPromptChunkRows, &untouched,
      &optimized_workspace, optimized_logits.data(), optimized_logits.size(),
      optimized_hidden.data(), optimized_hidden.size(), &cancelled_result,
      &cancel_control);
  status = reference.reset();
  bool cancellation_state_equal = false;
  if (status.is_ok()) status = untouched.state_equals(reference, &cancellation_state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool cancellation_outputs_unchanged = outputs_equal(
      optimized_logits, untouched_logits, optimized_hidden, untouched_hidden);
  const bool cancelled = cancel_status.code() == qw38::StatusCode::kCancelled &&
      cancellation.calls == cancellation.stop_after && untouched.frontier() == 0 &&
      cancellation_state_equal && cancellation_outputs_unchanged;
  std::printf("prompt_cancel=layer_boundary rows=4096 poll_calls=%zu status=%s "
              "frontier=%zu state_exact=%s outputs_unchanged=%s passed=%s\n",
              cancellation.calls, qw38::status_code_name(cancel_status.code()),
              untouched.frontier(), cancellation_state_equal ? "true" : "false",
              cancellation_outputs_unchanged ? "true" : "false",
              cancelled ? "true" : "false");

  qw38::cuda::SchedulerWorkspace tiny_workspace;
  status = tiny_workspace.create(3);
  if (!status.is_ok()) return fail_status(status);
  constexpr std::size_t kTinyBytes = 161595680;
  constexpr std::size_t kSmallBytes = 186711136;
  constexpr std::size_t kProductionBytes = 1819620960;
  const bool allocation_passed = tiny_workspace.allocated_bytes() == kTinyBytes &&
      small_workspace.allocated_bytes() == kSmallBytes &&
      optimized_workspace.allocated_bytes() == kProductionBytes;
  std::printf("prompt_workspace_bytes capacity_3=%zu capacity_65=%zu capacity_4097=%zu "
              "passed=%s\n", tiny_workspace.allocated_bytes(),
              small_workspace.allocated_bytes(), optimized_workspace.allocated_bytes(),
              allocation_passed ? "true" : "false");

  const bool passed = outer_passed && small_passed && bounds_passed &&
                      cancelled && allocation_passed;
  std::printf("status=%s\n", passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
