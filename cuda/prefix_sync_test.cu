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

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

bool expect_result(const char* name, const qw38::cuda::SyncResult& result,
                   std::size_t common, std::size_t reused,
                   std::size_t evaluated, bool replayed) {
  const bool passed = result.common_prefix == common &&
                      result.reused_tokens == reused &&
                      result.evaluated_tokens == evaluated &&
                      result.reset_and_replayed == replayed;
  std::printf(
      "prefix_case=%s common=%zu reused=%zu evaluated=%zu replayed=%s "
      "passed=%s\n",
      name, result.common_prefix, result.reused_tokens,
      result.evaluated_tokens, result.reset_and_replayed ? "true" : "false",
      passed ? "true" : "false");
  return passed;
}

qw38::Status sync(const qw38::cuda::ResidentModel& model,
                  const std::size_t* tokens, std::size_t count,
                  qw38::cuda::SchedulerSession* session,
                  qw38::cuda::SchedulerWorkspace* workspace,
                  std::vector<float>* logits,
                  std::array<float, qw38::internal::kResidualWidth>* hidden,
                  qw38::cuda::SyncResult* result) {
  return qw38::cuda::sync_tokens(
      model, tokens, count, session, workspace,
      count == 0 ? nullptr : logits->data(),
      count == 0 ? 0 : logits->size(), count == 0 ? nullptr : hidden->data(),
      count == 0 ? 0 : hidden->size(), result);
}

bool exact_outputs(const std::vector<float>& left_logits,
                   const std::vector<float>& right_logits,
                   const std::array<float, qw38::internal::kResidualWidth>&
                       left_hidden,
                   const std::array<float, qw38::internal::kResidualWidth>&
                       right_hidden) {
  return std::memcmp(left_logits.data(), right_logits.data(),
                     left_logits.size() * sizeof(float)) == 0 &&
         std::memcmp(left_hidden.data(), right_hidden.data(),
                     left_hidden.size() * sizeof(float)) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qw38-cuda-prefix-sync-test MODEL\n");
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
  qw38::cuda::SyncResult result;
  bool passed = true;

  constexpr std::array<std::size_t, 1> kFirst{42};
  status = sync(model, kFirst.data(), kFirst.size(), &primary,
                &primary_workspace, &primary_logits, &primary_hidden, &result);
  if (!status.is_ok()) return fail_status(status);
  passed = expect_result("initial", result, 0, 0, 1, false) && passed;

  constexpr std::array<std::size_t, 2> kAppended{42, 3649};
  status = sync(model, kAppended.data(), kAppended.size(), &primary,
                &primary_workspace, &primary_logits, &primary_hidden, &result);
  if (!status.is_ok()) return fail_status(status);
  passed = expect_result("append", result, 1, 1, 1, false) && passed;

  status = sync(model, kAppended.data(), kAppended.size(), &reference,
                &reference_workspace, &reference_logits, &reference_hidden,
                &result);
  if (!status.is_ok()) return fail_status(status);
  bool state_equal = false;
  status = primary.state_equals(reference, &state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool append_exact =
      state_equal && exact_outputs(primary_logits, reference_logits,
                                   primary_hidden, reference_hidden);
  const bool output_equal = exact_outputs(primary_logits, reference_logits,
                                          primary_hidden, reference_hidden);
  std::printf("prefix_exact=append_vs_fresh state=%s outputs=%s passed=%s\n",
              state_equal ? "true" : "false",
              output_equal ? "true" : "false",
              append_exact ? "true" : "false");
  passed = append_exact && passed;

  const std::vector<float> saved_logits = primary_logits;
  const auto saved_hidden = primary_hidden;
  status = sync(model, kAppended.data(), kAppended.size(), &primary,
                &primary_workspace, &primary_logits, &primary_hidden, &result);
  if (!status.is_ok()) return fail_status(status);
  const bool no_op_exact = exact_outputs(primary_logits, saved_logits,
                                         primary_hidden, saved_hidden);
  passed = expect_result("no_op", result, 2, 2, 0, false) && passed;
  std::printf("prefix_exact=no_op_outputs passed=%s\n",
              no_op_exact ? "true" : "false");
  passed = no_op_exact && passed;

  constexpr std::array<std::size_t, 2> kDiverged{42, 1219};
  status = sync(model, kDiverged.data(), kDiverged.size(), &primary,
                &primary_workspace, &primary_logits, &primary_hidden, &result);
  if (!status.is_ok()) return fail_status(status);
  passed = expect_result("divergent", result, 1, 0, 2, true) && passed;
  status = reference.reset();
  if (status.is_ok()) {
    status = sync(model, kDiverged.data(), kDiverged.size(), &reference,
                  &reference_workspace, &reference_logits, &reference_hidden,
                  &result);
  }
  if (!status.is_ok()) return fail_status(status);
  status = primary.state_equals(reference, &state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool divergent_exact =
      state_equal && exact_outputs(primary_logits, reference_logits,
                                   primary_hidden, reference_hidden);
  std::printf("prefix_exact=divergent_vs_fresh passed=%s\n",
              divergent_exact ? "true" : "false");
  passed = divergent_exact && passed;

  status = sync(model, kFirst.data(), kFirst.size(), &primary,
                &primary_workspace, &primary_logits, &primary_hidden, &result);
  if (!status.is_ok()) return fail_status(status);
  passed = expect_result("shorter", result, 1, 0, 1, true) && passed;
  status = reference.reset();
  if (status.is_ok()) {
    status = sync(model, kFirst.data(), kFirst.size(), &reference,
                  &reference_workspace, &reference_logits, &reference_hidden,
                  &result);
  }
  if (!status.is_ok()) return fail_status(status);
  status = primary.state_equals(reference, &state_equal);
  if (!status.is_ok()) return fail_status(status);
  const bool shorter_exact =
      state_equal && exact_outputs(primary_logits, reference_logits,
                                   primary_hidden, reference_hidden);
  std::printf("prefix_exact=shorter_vs_fresh passed=%s\n",
              shorter_exact ? "true" : "false");
  passed = shorter_exact && passed;

  constexpr std::array<std::size_t, 2> kInvalid{
      42, qw38::internal::kVocabularySize};
  status = sync(model, kInvalid.data(), kInvalid.size(), &primary,
                &primary_workspace, &primary_logits, &primary_hidden, &result);
  const bool rejected =
      status.code() == qw38::StatusCode::kInvalidArgument;
  status = primary.state_equals(reference, &state_equal);
  if (!status.is_ok()) return fail_status(status);
  std::printf("prefix_invalid=rejected_before_mutation rejected=%s "
              "state_equal=%s passed=%s\n",
              rejected ? "true" : "false", state_equal ? "true" : "false",
              rejected && state_equal ? "true" : "false");
  passed = rejected && state_equal && passed;

  qw38::cuda::SyncResult empty_result;
  status = sync(model, nullptr, 0, &primary, &primary_workspace,
                &primary_logits, &primary_hidden, &empty_result);
  if (!status.is_ok()) return fail_status(status);
  passed = expect_result("empty_reset", empty_result, 0, 0, 0, true) && passed;
  std::printf("prefix_run=complete frontier=%zu token_count=%zu passed=%s\n",
              primary.frontier(), primary.token_count(),
              passed ? "true" : "false");
  std::printf("status=%s\n", passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
