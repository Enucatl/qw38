#include "full_scheduler.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
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

qw38::Status eval(const qw38::cuda::ResidentModel& model, std::size_t token,
                  qw38::cuda::SchedulerSession* session,
                  qw38::cuda::SchedulerWorkspace* workspace,
                  std::vector<float>* logits,
                  std::array<float, qw38::internal::kResidualWidth>* hidden) {
  float elapsed = 0.0F;
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits->data(), logits->size(),
      hidden->data(), hidden->size(), &elapsed);
}

bool flip_byte(const std::string& source, const std::string& destination,
               std::streamoff offset) {
  std::error_code error;
  std::filesystem::copy_file(source, destination,
                             std::filesystem::copy_options::overwrite_existing,
                             error);
  if (error) return false;
  std::fstream file(destination, std::ios::binary | std::ios::in | std::ios::out);
  file.seekg(offset);
  char value = 0;
  file.read(&value, 1);
  value ^= 1;
  file.seekp(offset);
  file.write(&value, 1);
  file.close();
  return static_cast<bool>(file);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: qw38-cuda-checkpoint-test MODEL CHECKPOINT\n");
    return 1;
  }
  const std::string checkpoint = argv[2];
  const std::string corrupt = checkpoint + ".corrupt";
  const std::string incompatible = checkpoint + ".incompatible";
  std::remove(checkpoint.c_str());
  std::remove((checkpoint + ".tmp").c_str());
  std::remove(corrupt.c_str());
  std::remove(incompatible.c_str());

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
  qw38::cuda::SchedulerSession uninterrupted;
  qw38::cuda::SchedulerSession restored;
  if (status.is_ok()) status = uninterrupted.create(3);
  if (status.is_ok()) status = restored.create(3);
  qw38::cuda::SchedulerWorkspace uninterrupted_workspace;
  qw38::cuda::SchedulerWorkspace restored_workspace;
  if (status.is_ok()) status = uninterrupted_workspace.create(3);
  if (status.is_ok()) status = restored_workspace.create(3);
  if (!status.is_ok()) return fail_status(status);

  std::vector<float> uninterrupted_logits(qw38::internal::kVocabularySize);
  std::vector<float> restored_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> uninterrupted_hidden{};
  std::array<float, qw38::internal::kResidualWidth> restored_hidden{};
  status = eval(model, 42, &uninterrupted, &uninterrupted_workspace,
                &uninterrupted_logits, &uninterrupted_hidden);
  if (status.is_ok()) {
    status = eval(model, 3649, &uninterrupted, &uninterrupted_workspace,
                  &uninterrupted_logits, &uninterrupted_hidden);
  }
  const qw38::cuda::SamplerState sampler{0.75F, 0.9F, 40, 123456789ULL,
                                         987654321ULL};
  if (status.is_ok()) status = uninterrupted.set_sampler_state(sampler);
  if (status.is_ok()) status = uninterrupted.save_checkpoint(checkpoint);
  if (!status.is_ok()) return fail_status(status);

  const bool published = std::filesystem::exists(checkpoint) &&
                         !std::filesystem::exists(checkpoint + ".tmp");
  const std::uintmax_t checkpoint_bytes = std::filesystem::file_size(checkpoint);
  status = restored.restore_checkpoint(checkpoint, &restored_workspace);
  if (!status.is_ok()) return fail_status(status);
  bool equal = false;
  status = uninterrupted.state_equals(restored, &equal);
  if (!status.is_ok()) return fail_status(status);
  std::printf("checkpoint_case=round_trip bytes=%ju frontier=%zu "
              "published=%s state_equal=%s passed=%s\n",
              checkpoint_bytes, restored.frontier(),
              published ? "true" : "false", equal ? "true" : "false",
              published && equal ? "true" : "false");
  bool passed = published && equal;

  status = eval(model, 1277, &uninterrupted, &uninterrupted_workspace,
                &uninterrupted_logits, &uninterrupted_hidden);
  if (status.is_ok()) {
    status = eval(model, 1277, &restored, &restored_workspace, &restored_logits,
                  &restored_hidden);
  }
  if (!status.is_ok()) return fail_status(status);
  status = uninterrupted.state_equals(restored, &equal);
  if (!status.is_ok()) return fail_status(status);
  std::printf("checkpoint_case=exact_continuation frontier=%zu state_equal=%s "
              "passed=%s\n",
              restored.frontier(), equal ? "true" : "false",
              equal ? "true" : "false");
  passed = equal && passed;

  if (!flip_byte(checkpoint, corrupt, 248 + 17)) {
    std::fprintf(stderr, "cannot create corrupt checkpoint fixture\n");
    return 1;
  }
  status = restored.restore_checkpoint(corrupt, &restored_workspace);
  const bool corrupt_rejected =
      status.code() == qw38::StatusCode::kIncompatibleArtifact;
  status = uninterrupted.state_equals(restored, &equal);
  if (!status.is_ok()) return fail_status(status);
  std::printf("checkpoint_case=corrupt_rejected state_equal=%s passed=%s\n",
              equal ? "true" : "false",
              corrupt_rejected && equal ? "true" : "false");
  passed = corrupt_rejected && equal && passed;

  if (!flip_byte(checkpoint, incompatible, 80)) {
    std::fprintf(stderr, "cannot create incompatible checkpoint fixture\n");
    return 1;
  }
  status = restored.restore_checkpoint(incompatible, &restored_workspace);
  const bool incompatible_rejected =
      status.code() == qw38::StatusCode::kIncompatibleArtifact;
  status = uninterrupted.state_equals(restored, &equal);
  if (!status.is_ok()) return fail_status(status);
  std::printf("checkpoint_case=incompatible_rejected state_equal=%s "
              "passed=%s\n",
              equal ? "true" : "false",
              incompatible_rejected && equal ? "true" : "false");
  passed = incompatible_rejected && equal && passed;

  std::remove(checkpoint.c_str());
  std::remove(corrupt.c_str());
  std::remove(incompatible.c_str());
  std::printf("checkpoint_run=complete frontier=%zu passed=%s\n",
              restored.frontier(), passed ? "true" : "false");
  std::printf("status=%s\n", passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
