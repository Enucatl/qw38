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
  qw38::StatusCode code = qw38::StatusCode::kCancelled;
};

qw38::Status poll(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->calls != context->stop_after) return qw38::Status::ok();
  return {context->code, context->code == qw38::StatusCode::kCancelled
                             ? "diagnostic cancellation"
                             : "diagnostic injected error"};
}

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

qw38::Status eval(const qw38::cuda::ResidentModel& model, std::size_t token,
                  qw38::cuda::SchedulerSession* session,
                  qw38::cuda::SchedulerWorkspace* workspace,
                  std::vector<float>* logits,
                  std::array<float, qw38::internal::kResidualWidth>* hidden,
                  const qw38::cuda::EvalControl* control = nullptr) {
  float elapsed = 0.0F;
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits->data(), logits->size(),
      hidden->data(), hidden->size(), &elapsed, control);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qw38-cuda-atomic-eval-test MODEL\n");
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
  if (!status.is_ok()) return fail_status(status);

  qw38::cuda::ResidentModel model;
  status = model.upload(weights, mapping.data(), mapping.size());
  qw38::cuda::SchedulerSession primary;
  qw38::cuda::SchedulerSession reference;
  if (status.is_ok()) status = primary.create(3);
  if (status.is_ok()) status = reference.create(3);
  qw38::cuda::SchedulerWorkspace primary_workspace;
  qw38::cuda::SchedulerWorkspace reference_workspace;
  if (status.is_ok()) status = primary_workspace.create(3);
  if (status.is_ok()) status = reference_workspace.create(3);
  if (!status.is_ok()) return fail_status(status);

  std::vector<float> primary_logits(qw38::internal::kVocabularySize);
  std::vector<float> reference_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> primary_hidden{};
  std::array<float, qw38::internal::kResidualWidth> reference_hidden{};
  status = eval(model, 42, &primary, &primary_workspace, &primary_logits,
                &primary_hidden);
  if (status.is_ok()) {
    status = eval(model, 42, &reference, &reference_workspace,
                  &reference_logits, &reference_hidden);
  }
  if (!status.is_ok()) return fail_status(status);

  bool passed = true;
  bool equal = false;
  const std::vector<float> committed_logits = primary_logits;
  const auto committed_hidden = primary_hidden;
  PollContext cancellation{0, 8, qw38::StatusCode::kCancelled};
  const qw38::cuda::EvalControl cancel_control{poll, &cancellation};
  status = eval(model, 3649, &primary, &primary_workspace, &primary_logits,
                &primary_hidden, &cancel_control);
  const bool cancelled = status.code() == qw38::StatusCode::kCancelled;
  status = primary.state_equals(reference, &equal);
  if (!status.is_ok()) return fail_status(status);
  const bool cancellation_outputs_unchanged =
      std::memcmp(primary_logits.data(), committed_logits.data(),
                  primary_logits.size() * sizeof(float)) == 0 &&
      std::memcmp(primary_hidden.data(), committed_hidden.data(),
                  primary_hidden.size() * sizeof(float)) == 0;
  std::printf("atomic_case=cancellation poll_calls=%zu frontier=%zu "
              "state_equal=%s outputs_unchanged=%s passed=%s\n",
              cancellation.calls, primary.frontier(), equal ? "true" : "false",
              cancellation_outputs_unchanged ? "true" : "false",
              cancelled && equal && cancellation_outputs_unchanged &&
                      primary.frontier() == 1
                  ? "true"
                  : "false");
  passed = cancelled && equal && cancellation_outputs_unchanged &&
           primary.frontier() == 1 && passed;

  PollContext failure{0, 31, qw38::StatusCode::kInternal};
  const qw38::cuda::EvalControl failure_control{poll, &failure};
  status = eval(model, 3649, &primary, &primary_workspace, &primary_logits,
                &primary_hidden, &failure_control);
  const bool failed = status.code() == qw38::StatusCode::kInternal;
  status = primary.state_equals(reference, &equal);
  if (!status.is_ok()) return fail_status(status);
  const bool failure_outputs_unchanged =
      std::memcmp(primary_logits.data(), committed_logits.data(),
                  primary_logits.size() * sizeof(float)) == 0 &&
      std::memcmp(primary_hidden.data(), committed_hidden.data(),
                  primary_hidden.size() * sizeof(float)) == 0;
  std::printf("atomic_case=injected_error poll_calls=%zu frontier=%zu "
              "state_equal=%s outputs_unchanged=%s passed=%s\n",
              failure.calls, primary.frontier(), equal ? "true" : "false",
              failure_outputs_unchanged ? "true" : "false",
              failed && equal && failure_outputs_unchanged &&
                      primary.frontier() == 1
                  ? "true"
                  : "false");
  passed = failed && equal && failure_outputs_unchanged &&
           primary.frontier() == 1 && passed;

  status = eval(model, 3649, &primary, &primary_workspace, &primary_logits,
                &primary_hidden);
  if (status.is_ok()) {
    status = eval(model, 3649, &reference, &reference_workspace,
                  &reference_logits, &reference_hidden);
  }
  if (!status.is_ok()) return fail_status(status);
  status = primary.state_equals(reference, &equal);
  if (!status.is_ok()) return fail_status(status);
  std::printf("atomic_case=successful_commit frontier=%zu state_equal=%s "
              "passed=%s\n",
              primary.frontier(), equal ? "true" : "false",
              equal && primary.frontier() == 2 ? "true" : "false");
  passed = equal && primary.frontier() == 2 && passed;

  std::size_t sampled = 0;
  status = qw38::cuda::greedy_sample(primary, &sampled);
  if (!status.is_ok()) return fail_status(status);
  status = primary.state_equals(reference, &equal);
  if (!status.is_ok()) return fail_status(status);
  const bool sampling_pure = sampled == 1277 && equal && primary.frontier() == 2;
  std::printf("atomic_case=separate_sampling token=%zu frontier=%zu "
              "state_equal=%s passed=%s\n",
              sampled, primary.frontier(), equal ? "true" : "false",
              sampling_pure ? "true" : "false");
  passed = sampling_pure && passed;

  status = eval(model, qw38::internal::kVocabularySize, &primary,
                &primary_workspace, &primary_logits, &primary_hidden);
  const bool invalid = status.code() == qw38::StatusCode::kInvalidArgument;
  status = primary.state_equals(reference, &equal);
  if (!status.is_ok()) return fail_status(status);
  std::printf("atomic_case=preflight_error frontier=%zu state_equal=%s "
              "passed=%s\n",
              primary.frontier(), equal ? "true" : "false",
              invalid && equal ? "true" : "false");
  passed = invalid && equal && passed;

  std::printf("atomic_run=complete workspace_bytes=%zu frontier=%zu passed=%s\n",
              primary_workspace.allocated_bytes(), primary.frontier(),
              passed ? "true" : "false");
  std::printf("status=%s\n", passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
