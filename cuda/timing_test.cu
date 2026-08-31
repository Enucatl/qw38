#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "full_scheduler.h"
#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

void print_value(const char* name, const qw38::cuda::TimingValue& value) {
  if (value.measured) {
    std::printf("timing_category=%s availability=measured milliseconds=%.9g\n",
                name, value.milliseconds);
  } else {
    std::printf("timing_category=%s availability=unavailable milliseconds=null\n",
                name);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf CHECKPOINT\n", argv[0]);
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
  if (status.is_ok()) status = model.upload(weights, mapping.data(), mapping.size());
  qw38::cuda::SchedulerSession session;
  if (status.is_ok()) status = session.create(2);
  qw38::cuda::SchedulerWorkspace workspace;
  if (status.is_ok()) status = workspace.create(2);
  if (!status.is_ok()) return fail_status(status);

  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  float compute_ms = 0.0F;
  status = qw38::cuda::execute_token(
      model, 42, &session, &workspace, logits.data(), logits.size(),
      hidden.data(), hidden.size(), &compute_ms);
  qw38::cuda::RuntimeTimings timings;
  if (status.is_ok()) {
    status = qw38::cuda::execute_token(
        model, 3649, &session, &workspace, logits.data(), logits.size(),
        hidden.data(), hidden.size(), &compute_ms, nullptr, &timings);
  }
  std::size_t sampled = 0;
  if (status.is_ok()) status = qw38::cuda::greedy_sample(session, &sampled, &timings);
  if (status.is_ok()) status = session.save_checkpoint(argv[2], &timings);
  if (!status.is_ok()) return fail_status(status);

  print_value("loading", timings.loading);
  print_value("embedding", timings.embedding);
  print_value("gdn", timings.gdn);
  print_value("attention", timings.attention);
  print_value("ffn", timings.ffn);
  print_value("logits", timings.logits);
  print_value("sampling", timings.sampling);
  print_value("graph_launch", timings.graph_launch);
  print_value("queueing", timings.queueing);
  print_value("persistence", timings.persistence);
  print_value("idle_gaps", timings.idle_gaps);
  print_value("state_commit", timings.state_commit);
  print_value("token_total", timings.token_total);
  const float attributed = timings.embedding.milliseconds +
                           timings.gdn.milliseconds +
                           timings.attention.milliseconds +
                           timings.ffn.milliseconds +
                           timings.logits.milliseconds +
                           timings.state_commit.milliseconds +
                           timings.idle_gaps.milliseconds;
  std::printf(
      "timing_run=token_1 sampled=%zu compute_ms=%.9g attributed_ms=%.9g "
      "total_ms=%.9g frontier=%zu\n",
      sampled, compute_ms, attributed, timings.token_total.milliseconds,
      session.frontier());
  std::printf("status=passed\n");
  std::remove(argv[2]);
  return 0;
}
