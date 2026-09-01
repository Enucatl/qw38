#include "full_scheduler.h"

#include <algorithm>
#include <array>
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

qw38::Status run_token(const qw38::cuda::ResidentModel& model, Path* path,
                       qw38::cuda::SchedulerGraphs* graphs, float* elapsed,
                       qw38::cuda::RuntimeTimings* timings = nullptr) {
  return qw38::cuda::execute_token(
      model, 42, &path->session, &path->workspace, path->logits.data(),
      path->logits.size(), path->hidden.data(), path->hidden.size(), elapsed,
      nullptr, timings, qw38::cuda::PointwisePath::kFused, graphs);
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
  Path graphed;
  Path ordinary;
  graphed.logits.resize(qw38::internal::kVocabularySize);
  ordinary.logits.resize(qw38::internal::kVocabularySize);
  graphed.samples.reserve(kSamples);
  ordinary.samples.reserve(kSamples);
  if (status.is_ok()) status = graphed.session.create(kCapacity);
  if (status.is_ok()) status = graphed.workspace.create(kCapacity);
  if (status.is_ok()) status = ordinary.session.create(kCapacity);
  if (status.is_ok()) status = ordinary.workspace.create(kCapacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &graphed.workspace);
  if (!status.is_ok()) return fail_status(status);

  float invalid_ms = 0.0F;
  const qw38::Status mismatch = run_token(model, &ordinary, &graphs, &invalid_ms);
  if (mismatch.code() != qw38::StatusCode::kInvalidArgument ||
      ordinary.session.frontier() != 0) {
    std::fprintf(stderr, "mismatched graph addresses did not fail closed\n");
    return 1;
  }

  bool exact_logits = true;
  bool exact_hidden = true;
  float graph_launch_ms = 0.0F;
  for (std::size_t sample = 0; status.is_ok() && sample < kCapacity; ++sample) {
    float graph_ms = 0.0F;
    float ordinary_ms = 0.0F;
    qw38::cuda::RuntimeTimings timings;
    qw38::cuda::RuntimeTimings* timing_pointer = sample == 0 ? &timings : nullptr;
    if (sample % 2 == 0) {
      status = run_token(model, &graphed, &graphs, &graph_ms, timing_pointer);
      if (status.is_ok()) {
        status = run_token(model, &ordinary, nullptr, &ordinary_ms);
      }
    } else {
      status = run_token(model, &ordinary, nullptr, &ordinary_ms);
      if (status.is_ok()) {
        status = run_token(model, &graphed, &graphs, &graph_ms, timing_pointer);
      }
    }
    if (!status.is_ok()) break;
    if (sample == 0) {
      if (!timings.graph_launch.measured) {
        std::fprintf(stderr, "graph launch attribution was not measured\n");
        return 1;
      }
      graph_launch_ms = timings.graph_launch.milliseconds;
    }
    exact_logits = exact_logits &&
                   std::memcmp(graphed.logits.data(), ordinary.logits.data(),
                               graphed.logits.size() * sizeof(float)) == 0;
    exact_hidden = exact_hidden &&
                   std::memcmp(graphed.hidden.data(), ordinary.hidden.data(),
                               graphed.hidden.size() * sizeof(float)) == 0;
    if (sample >= kWarmups) {
      graphed.samples.push_back(graph_ms);
      ordinary.samples.push_back(ordinary_ms);
    }
  }
  if (!status.is_ok()) return fail_status(status);

  constexpr std::size_t kTapValues = 4 * qw38::internal::kResidualWidth;
  std::vector<float> graph_taps(kTapValues);
  std::vector<float> ordinary_taps(kTapValues);
  status = graphed.workspace.copy_trace_taps(graph_taps.data(),
                                             graph_taps.size());
  if (status.is_ok()) {
    status = ordinary.workspace.copy_trace_taps(ordinary_taps.data(),
                                                ordinary_taps.size());
  }
  bool exact_state = false;
  if (status.is_ok()) {
    status = graphed.session.state_equals(ordinary.session, &exact_state);
  }
  std::size_t graph_greedy = 0;
  std::size_t ordinary_greedy = 0;
  if (status.is_ok()) {
    status = qw38::cuda::greedy_sample(graphed.session, &graph_greedy);
  }
  if (status.is_ok()) {
    status = qw38::cuda::greedy_sample(ordinary.session, &ordinary_greedy);
  }
  if (!status.is_ok()) return fail_status(status);

  const bool exact_taps =
      std::memcmp(graph_taps.data(), ordinary_taps.data(),
                  graph_taps.size() * sizeof(float)) == 0;
  float graph_sum = 0.0F;
  float ordinary_sum = 0.0F;
  for (float value : graphed.samples) graph_sum += value;
  for (float value : ordinary.samples) ordinary_sum += value;
  const float graph_mean = graph_sum / static_cast<float>(kSamples);
  const float ordinary_mean = ordinary_sum / static_cast<float>(kSamples);
  for (std::size_t sample = 0; sample < kSamples; ++sample) {
    const std::size_t execution_index = sample + kWarmups;
    std::printf(
        "graph_sample=%zu graph_ms=%.9g ordinary_ms=%.9g first=%s\n", sample,
        graphed.samples[sample], ordinary.samples[sample],
        execution_index % 2 == 0 ? "graph" : "ordinary");
  }
  std::printf(
      "graph_compare=ffn exact_logits=%s exact_hidden=%s exact_taps=%s "
      "exact_state=%s greedy_equal=%s frontier=%zu\n",
      exact_logits ? "true" : "false", exact_hidden ? "true" : "false",
      exact_taps ? "true" : "false", exact_state ? "true" : "false",
      graph_greedy == ordinary_greedy ? "true" : "false",
      graphed.session.frontier());
  std::printf(
      "graph_run=ffn graphs=%zu graph_bytes=%zu graph_launch_cpu_ms=%.9g "
      "warmups=%zu samples=%zu graph_mean_ms=%.9g ordinary_mean_ms=%.9g "
      "graph_median_ms=%.9g ordinary_median_ms=%.9g speedup=%.9g\n",
      graphs.graph_count(), graphs.allocated_bytes(), graph_launch_ms, kWarmups,
      kSamples, graph_mean, ordinary_mean, median(graphed.samples),
      median(ordinary.samples), ordinary_mean / graph_mean);
  const bool passed = exact_logits && exact_hidden && exact_taps && exact_state &&
                      graph_greedy == ordinary_greedy;
  std::printf("status=%s\n", passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
