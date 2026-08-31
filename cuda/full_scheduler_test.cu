#include "full_scheduler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#include "model.h"
#include "scalar_runtime.h"
#include "scheduler.h"
#include "weights.h"

#include "diagnostic_trace.h"

namespace {

struct Metrics final {
  float maximum_absolute = 0.0F;
  float maximum_relative = 0.0F;
  double squared = 0.0;
  double dot = 0.0;
  double actual_squared = 0.0;
  double expected_squared = 0.0;
  std::size_t count = 0;
  std::size_t nonfinite = 0;
  std::size_t first_over_absolute = std::numeric_limits<std::size_t>::max();
};

constexpr std::array<std::size_t, 3> kTraceLayers{0, 3, 63};
constexpr std::size_t kTraceRows = 4;

struct ScalarTapCapture final {
  float* output;
};

qw38::Status capture_scalar_tap(
    const qw38::internal::TraceTensorView& tensor, void* context) noexcept {
  auto* capture = static_cast<ScalarTapCapture*>(context);
  if (std::strcmp(tensor.name, "layer_residual") != 0 ||
      tensor.value_count != qw38::internal::kResidualWidth) {
    return qw38::Status::ok();
  }
  for (std::size_t slot = 0; slot < kTraceLayers.size(); ++slot) {
    if (tensor.layer == kTraceLayers[slot]) {
      std::copy(tensor.values, tensor.values + tensor.value_count,
                capture->output + slot * tensor.value_count);
    }
  }
  return qw38::Status::ok();
}

bool passes(const Metrics& metrics, float maximum_absolute,
            double maximum_rms, double minimum_cosine) {
  const double rms = std::sqrt(metrics.squared / metrics.count);
  const double denominator =
      std::sqrt(metrics.actual_squared * metrics.expected_squared);
  const double cosine = denominator == 0.0 ? 1.0 : metrics.dot / denominator;
  return metrics.nonfinite == 0 &&
         metrics.maximum_absolute <= maximum_absolute &&
         rms <= maximum_rms && cosine >= minimum_cosine;
}

Metrics measure(const float* actual, const float* expected, std::size_t count,
                float absolute_gate) {
  Metrics metrics;
  metrics.count = count;
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      ++metrics.nonfinite;
      continue;
    }
    const float error = std::fabs(actual[index] - expected[index]);
    const float relative = error / std::max(std::fabs(expected[index]), 1.0e-6F);
    metrics.maximum_absolute = std::max(metrics.maximum_absolute, error);
    metrics.maximum_relative = std::max(metrics.maximum_relative, relative);
    metrics.squared += static_cast<double>(error) * error;
    metrics.dot += static_cast<double>(actual[index]) * expected[index];
    metrics.actual_squared +=
        static_cast<double>(actual[index]) * actual[index];
    metrics.expected_squared +=
        static_cast<double>(expected[index]) * expected[index];
    if (error > absolute_gate &&
        metrics.first_over_absolute == std::numeric_limits<std::size_t>::max()) {
      metrics.first_over_absolute = index;
    }
  }
  return metrics;
}

void print_metrics(const char* boundary, std::size_t row,
                   const Metrics& metrics) {
  const double rms = std::sqrt(metrics.squared / metrics.count);
  const double denominator =
      std::sqrt(metrics.actual_squared * metrics.expected_squared);
  const double cosine = denominator == 0.0 ? 1.0 : metrics.dot / denominator;
  std::printf(
      "scheduler_compare=%s row=%zu count=%zu max_abs=%.9g max_rel=%.9g "
      "rms=%.12g cosine=%.12g nonfinite=%zu first_over_absolute=",
      boundary, row, metrics.count, metrics.maximum_absolute,
      metrics.maximum_relative, rms, cosine, metrics.nonfinite);
  if (metrics.first_over_absolute == std::numeric_limits<std::size_t>::max()) {
    std::printf("none\n");
  } else {
    std::printf("%zu\n", metrics.first_over_absolute);
  }
}

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(
        stderr,
        "usage: qw38-cuda-full-scheduler-test MODEL LLAMA_LOGITS\n");
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

  std::vector<float> llama_logits(2 * qw38::internal::kVocabularySize);
  std::ifstream llama_input(argv[2], std::ios::binary);
  llama_input.read(reinterpret_cast<char*>(llama_logits.data()),
                   static_cast<std::streamsize>(llama_logits.size() *
                                                sizeof(float)));
  if (!llama_input || llama_input.peek() != std::ifstream::traits_type::eof()) {
    std::fprintf(stderr, "pinned llama.cpp logits have the wrong size\n");
    return 1;
  }

  qw38::cuda::ResidentModel model;
  status = model.upload(weights, mapping.data(), mapping.size());
  qw38::cuda::SchedulerSession session;
  if (status.is_ok()) status = session.create(2);
  qw38::cuda::SchedulerWorkspace workspace;
  if (status.is_ok()) status = workspace.create(2);
  if (!status.is_ok()) return fail_status(status);

  constexpr std::array<std::size_t, 2> kTokens{42, 3649};
  std::vector<float> cuda_logits(kTokens.size() * qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> cuda_hidden{};
  std::array<float, kTokens.size()> cuda_ms{};
  std::vector<float> cuda_taps(kTokens.size() * kTraceRows *
                               qw38::internal::kResidualWidth);
  for (std::size_t row = 0; row < kTokens.size(); ++row) {
    status = qw38::cuda::execute_token(
        model, kTokens[row], &session, &workspace,
        cuda_logits.data() + row * qw38::internal::kVocabularySize,
        qw38::internal::kVocabularySize, cuda_hidden.data(),
        cuda_hidden.size(), &cuda_ms[row]);
    if (!status.is_ok()) return fail_status(status);
    status = workspace.copy_trace_taps(
        cuda_taps.data() +
            row * kTraceRows * qw38::internal::kResidualWidth,
        kTraceRows * qw38::internal::kResidualWidth);
    if (!status.is_ok()) return fail_status(status);
  }

  qw38::internal::ScalarModelParameters parameters;
  status = qw38::internal::prepare_scalar_model_parameters(weights, &parameters);
  qw38::internal::ScalarSessionState scalar_state;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_session_state(2, &scalar_state);
  }
  qw38::internal::ScalarWorkspace scalar_workspace;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_workspace(2, &scalar_workspace);
  }
  std::vector<float> scalar_logits(kTokens.size() *
                                   qw38::internal::kVocabularySize);
  std::vector<float> scalar_taps(kTokens.size() * kTraceRows *
                                 qw38::internal::kResidualWidth);
  const auto scalar_start = std::chrono::steady_clock::now();
  for (std::size_t row = 0; status.is_ok() && row < kTokens.size(); ++row) {
    ScalarTapCapture capture{
        scalar_taps.data() +
        row * kTraceRows * qw38::internal::kResidualWidth};
    const qw38::internal::TraceFilter filter{
        qw38::internal::kTraceAllLayers, "layer_residual"};
    status = qw38::internal::execute_scalar_token_traced(
        weights, parameters, kTokens[row], &scalar_state, &scalar_workspace,
        scalar_logits.data() + row * qw38::internal::kVocabularySize,
        qw38::internal::kVocabularySize, filter, capture_scalar_tap, &capture);
    if (status.is_ok()) {
      std::copy(scalar_workspace.final_normalized.begin(),
                scalar_workspace.final_normalized.end(),
                capture.output + 3 * qw38::internal::kResidualWidth);
    }
  }
  const double scalar_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - scalar_start)
                               .count();
  if (!status.is_ok()) return fail_status(status);

  bool greedy_equal = true;
  bool scalar_admitted = true;
  bool llama_admitted = true;
  for (std::size_t row = 0; row < kTokens.size(); ++row) {
    const float* actual = cuda_logits.data() +
                          row * qw38::internal::kVocabularySize;
    const float* expected = scalar_logits.data() +
                            row * qw38::internal::kVocabularySize;
    const Metrics metrics = measure(actual, expected,
                                    qw38::internal::kVocabularySize, 0.21F);
    print_metrics("logits", row, metrics);
    const std::size_t actual_greedy = static_cast<std::size_t>(
        std::max_element(actual, actual + qw38::internal::kVocabularySize) -
        actual);
    const std::size_t expected_greedy = static_cast<std::size_t>(
        std::max_element(expected, expected + qw38::internal::kVocabularySize) -
        expected);
    std::printf("scheduler_greedy=row_%zu cuda=%zu scalar=%zu equal=%s\n", row,
                actual_greedy, expected_greedy,
                actual_greedy == expected_greedy ? "true" : "false");
    greedy_equal = greedy_equal && actual_greedy == expected_greedy;
    scalar_admitted = scalar_admitted &&
                      passes(metrics, 0.21F, 0.037, 0.999831);

    const float* llama = llama_logits.data() +
                         row * qw38::internal::kVocabularySize;
    const Metrics llama_metrics = measure(
        actual, llama, qw38::internal::kVocabularySize, 0.21F);
    print_metrics("llama_logits", row, llama_metrics);
    const double llama_rms =
        std::sqrt(llama_metrics.squared / llama_metrics.count);
    const double llama_denominator = std::sqrt(
        llama_metrics.actual_squared * llama_metrics.expected_squared);
    const double llama_cosine = llama_denominator == 0.0
                                    ? 1.0
                                    : llama_metrics.dot / llama_denominator;
    llama_admitted = llama_admitted && llama_metrics.nonfinite == 0 &&
                     llama_metrics.maximum_absolute <= 0.21F &&
                     llama_rms <= 0.037 && llama_cosine >= 0.999831;
  }
  const Metrics hidden = measure(cuda_hidden.data(),
                                 scalar_workspace.activation_a.data(),
                                 cuda_hidden.size(), 0.21F);
  print_metrics("final_hidden", 1, hidden);

  constexpr std::array<float, kTraceRows> kTapAbsolute{0.012F, 0.04F, 1.7F,
                                                        0.21F};
  constexpr std::array<double, kTraceRows> kTapRms{0.0012, 0.0029, 0.16,
                                                   0.035};
  constexpr std::array<double, kTraceRows> kTapCosine{0.999989, 0.999985,
                                                      0.999854, 0.999852};
  constexpr std::array<const char*, kTraceRows> kTapNames{
      "layer_0_residual", "layer_3_residual", "layer_63_residual",
      "final_norm"};
  bool taps_admitted = true;
  for (std::size_t row = 0; row < kTokens.size(); ++row) {
    for (std::size_t tap = 0; tap < kTraceRows; ++tap) {
      const std::size_t offset =
          (row * kTraceRows + tap) * qw38::internal::kResidualWidth;
      const Metrics tap_metrics = measure(
          cuda_taps.data() + offset, scalar_taps.data() + offset,
          qw38::internal::kResidualWidth, kTapAbsolute[tap]);
      print_metrics(kTapNames[tap], row, tap_metrics);
      taps_admitted = taps_admitted &&
                      passes(tap_metrics, kTapAbsolute[tap], kTapRms[tap],
                             kTapCosine[tap]);
    }
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t memory_error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (memory_error != cudaSuccess) {
    std::fprintf(stderr, "cudaMemGetInfo: %s\n",
                 cudaGetErrorString(memory_error));
    return 1;
  }
  std::printf(
      "scheduler_run=full layers=64 gdn_layers=48 attention_layers=16 "
      "frontier=%zu model_bytes=%zu session_bytes=%zu workspace_bytes=%zu "
      "upload_ms=%.6f token0_ms=%.6f token1_ms=%.6f scalar_ms=%.6f "
      "free_bytes=%zu total_bytes=%zu\n",
      session.frontier(), model.resident_bytes(), session.allocated_bytes(),
      workspace.allocated_bytes(), model.upload_milliseconds(), cuda_ms[0],
      cuda_ms[1], scalar_ms, free_bytes, total_bytes);
  std::printf("scheduler_admission=scalar passed=%s\n",
              scalar_admitted ? "true" : "false");
  std::printf("scheduler_admission=scalar_taps passed=%s\n",
              taps_admitted ? "true" : "false");
  std::printf("scheduler_admission=llama_same_gguf passed=%s\n",
              llama_admitted ? "true" : "false");
  std::printf("status=%s\n",
              scalar_admitted && taps_admitted && greedy_equal ? "passed"
                                                                : "failed");
  return scalar_admitted && taps_admitted && greedy_equal ? 0 : 1;
}
