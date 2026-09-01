#include "full_scheduler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr std::size_t kWarmups = 3;
constexpr std::size_t kSamples = 30;
constexpr std::size_t kCapacity = kWarmups + kSamples;

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return (values[values.size() / 2 - 1] + values[values.size() / 2]) * 0.5F;
}

struct Path final {
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  std::vector<float> logits;
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  std::vector<float> samples;
};

qw38::Status run_token(const qw38::cuda::ResidentModel& model,
                       qw38::cuda::PointwisePath pointwise_path,
                       Path* path, float* elapsed) {
  return qw38::cuda::execute_token(
      model, 42, &path->session, &path->workspace, path->logits.data(),
      path->logits.size(), path->hidden.data(), path->hidden.size(), elapsed,
      nullptr, nullptr, pointwise_path);
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
  if (status.is_ok()) status = model.upload(weights, mapping.data(), mapping.size());
  Path fused;
  Path unfused;
  fused.logits.resize(qw38::internal::kVocabularySize);
  unfused.logits.resize(qw38::internal::kVocabularySize);
  fused.samples.reserve(kSamples);
  unfused.samples.reserve(kSamples);
  if (status.is_ok()) status = fused.session.create(kCapacity);
  if (status.is_ok()) status = fused.workspace.create(kCapacity);
  if (status.is_ok()) status = unfused.session.create(kCapacity);
  if (status.is_ok()) status = unfused.workspace.create(kCapacity);
  if (!status.is_ok()) return fail_status(status);

  float invalid_ms = 0.0F;
  const qw38::Status invalid_status = qw38::cuda::execute_token(
      model, 42, &fused.session, &fused.workspace, fused.logits.data(),
      fused.logits.size(), fused.hidden.data(), fused.hidden.size(), &invalid_ms,
      nullptr, nullptr, static_cast<qw38::cuda::PointwisePath>(255));
  if (invalid_status.code() != qw38::StatusCode::kInvalidArgument ||
      fused.session.frontier() != 0) {
    std::fprintf(stderr, "invalid pointwise path did not fail before mutation\n");
    return 1;
  }

  bool exact_logits = true;
  bool exact_hidden = true;
  for (std::size_t sample = 0; status.is_ok() && sample < kCapacity; ++sample) {
    float fused_ms = 0.0F;
    float unfused_ms = 0.0F;
    if (sample % 2 == 0) {
      status = run_token(model, qw38::cuda::PointwisePath::kFused, &fused,
                         &fused_ms);
      if (status.is_ok()) {
        status = run_token(model, qw38::cuda::PointwisePath::kUnfused,
                           &unfused, &unfused_ms);
      }
    } else {
      status = run_token(model, qw38::cuda::PointwisePath::kUnfused, &unfused,
                         &unfused_ms);
      if (status.is_ok()) {
        status = run_token(model, qw38::cuda::PointwisePath::kFused, &fused,
                           &fused_ms);
      }
    }
    if (!status.is_ok()) break;
    exact_logits = exact_logits &&
                   std::memcmp(fused.logits.data(), unfused.logits.data(),
                               fused.logits.size() * sizeof(float)) == 0;
    exact_hidden = exact_hidden &&
                   std::memcmp(fused.hidden.data(), unfused.hidden.data(),
                               fused.hidden.size() * sizeof(float)) == 0;
    if (sample >= kWarmups) {
      fused.samples.push_back(fused_ms);
      unfused.samples.push_back(unfused_ms);
    }
  }
  if (!status.is_ok()) return fail_status(status);

  constexpr std::size_t kTapValues = 4 * qw38::internal::kResidualWidth;
  std::vector<float> fused_taps(kTapValues);
  std::vector<float> unfused_taps(kTapValues);
  status = fused.workspace.copy_trace_taps(fused_taps.data(), fused_taps.size());
  if (status.is_ok()) {
    status = unfused.workspace.copy_trace_taps(unfused_taps.data(),
                                               unfused_taps.size());
  }
  bool exact_state = false;
  if (status.is_ok()) status = fused.session.state_equals(unfused.session,
                                                          &exact_state);
  std::size_t fused_greedy = 0;
  std::size_t unfused_greedy = 0;
  if (status.is_ok()) status = qw38::cuda::greedy_sample(fused.session,
                                                         &fused_greedy);
  if (status.is_ok()) status = qw38::cuda::greedy_sample(unfused.session,
                                                         &unfused_greedy);
  if (!status.is_ok()) return fail_status(status);

  const bool exact_taps =
      std::memcmp(fused_taps.data(), unfused_taps.data(),
                  fused_taps.size() * sizeof(float)) == 0;
  float fused_sum = 0.0F;
  float unfused_sum = 0.0F;
  for (float value : fused.samples) fused_sum += value;
  for (float value : unfused.samples) unfused_sum += value;
  const float fused_mean = fused_sum / static_cast<float>(kSamples);
  const float unfused_mean = unfused_sum / static_cast<float>(kSamples);
  const float speedup = unfused_mean / fused_mean;
  for (std::size_t sample = 0; sample < kSamples; ++sample) {
    const std::size_t execution_index = sample + kWarmups;
    std::printf(
        "fusion_sample=%zu fused_ms=%.9g unfused_ms=%.9g first=%s\n", sample,
        fused.samples[sample], unfused.samples[sample],
        execution_index % 2 == 0 ? "fused" : "unfused");
  }
  std::printf(
      "fusion_compare=next_input_norm exact_logits=%s exact_hidden=%s "
      "exact_taps=%s exact_state=%s greedy_equal=%s frontier=%zu\n",
      exact_logits ? "true" : "false", exact_hidden ? "true" : "false",
      exact_taps ? "true" : "false", exact_state ? "true" : "false",
      fused_greedy == unfused_greedy ? "true" : "false",
      fused.session.frontier());
  std::printf(
      "fusion_timing=next_input_norm warmups=%zu samples=%zu fused_mean_ms=%.9g "
      "unfused_mean_ms=%.9g fused_median_ms=%.9g unfused_median_ms=%.9g "
      "speedup=%.9g fused_faster=%s\n",
      kWarmups, kSamples, fused_mean, unfused_mean, median(fused.samples),
      median(unfused.samples), speedup, fused_mean < unfused_mean ? "true"
                                                              : "false");
  const bool passed = exact_logits && exact_hidden && exact_taps && exact_state &&
                      fused_greedy == unfused_greedy;
  std::printf("status=%s\n", passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
