#include "quant_mmv.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kWarmups = 3;
constexpr int kSamples = 30;

std::size_t weight_bytes(qw38::cuda::QuantKind kind, std::size_t rows,
                         std::size_t columns) {
  const std::size_t values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? 32 : 256;
  const std::size_t bytes =
      kind == qw38::cuda::QuantKind::kQ4K ? 144
      : kind == qw38::cuda::QuantKind::kQ6K ? 210
                                            : 34;
  return rows * (columns / values) * bytes;
}

const char* kind_name(qw38::cuda::QuantKind kind) {
  if (kind == qw38::cuda::QuantKind::kQ4K) return "q4_k";
  if (kind == qw38::cuda::QuantKind::kQ6K) return "q6_k";
  return "q8_0";
}

int fail(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

int run_mmv(qw38::cuda::QuantKind kind, std::size_t rows,
            std::size_t columns) {
  std::uint8_t* weights = nullptr;
  __nv_bfloat16* activation = nullptr;
  qw38::cuda::Q8Block* staged = nullptr;
  float* output = nullptr;
  cudaError_t error = cudaMalloc(&weights, weight_bytes(kind, rows, columns));
  if (error == cudaSuccess) {
    error = cudaMalloc(&activation, columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&staged, qw38::cuda::q8_workspace_bytes(columns));
  }
  if (error == cudaSuccess) error = cudaMalloc(&output, rows * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMemset(weights, 0, weight_bytes(kind, rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(activation, 0, columns * sizeof(__nv_bfloat16));
  }
  if (error != cudaSuccess) return fail("MMV allocation", error);

  for (unsigned int warps : {4U, 8U, 16U}) {
    for (int warmup = 0; warmup < kWarmups && error == cudaSuccess; ++warmup) {
      error = qw38::cuda::launch_quant_mmv_variant(
          kind, weights, rows, columns, activation, staged, output, warps,
          nullptr);
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (error == cudaSuccess) error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    for (int sample = 0; sample < kSamples && error == cudaSuccess; ++sample) {
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_quant_mmv_variant(
            kind, weights, rows, columns, activation, staged, output, warps,
            nullptr);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float milliseconds = 0.0F;
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&milliseconds, start, stop);
      }
      if (error == cudaSuccess) {
        std::printf(
            "tune=mmv kind=%s rows=%zu columns=%zu warps=%u sample=%d "
            "milliseconds=%.9g\n",
            kind_name(kind), rows, columns, warps, sample, milliseconds);
      }
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    std::array<float, 1> value{};
    error = cudaMemcpy(value.data(), output, sizeof(float), cudaMemcpyDeviceToHost);
    if (error == cudaSuccess && value[0] != 0.0F) error = cudaErrorUnknown;
  }
  cudaFree(output);
  cudaFree(staged);
  cudaFree(activation);
  cudaFree(weights);
  return error == cudaSuccess ? 0 : fail("MMV sweep", error);
}

int run_mmq(std::size_t prompt_rows) {
  constexpr std::size_t kRows = 17408;
  constexpr std::size_t kColumns = 5120;
  constexpr qw38::cuda::QuantKind kKind = qw38::cuda::QuantKind::kQ4K;
  std::uint8_t* weights = nullptr;
  __nv_bfloat16* prompt = nullptr;
  qw38::cuda::Q8Block* staged = nullptr;
  float* output = nullptr;
  cudaError_t error = cudaMalloc(&weights, weight_bytes(kKind, kRows, kColumns));
  if (error == cudaSuccess) {
    error = cudaMalloc(&prompt,
                       prompt_rows * kColumns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &staged,
        qw38::cuda::q8_prompt_workspace_bytes(prompt_rows, kColumns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&output, prompt_rows * kRows * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(weights, 0, weight_bytes(kKind, kRows, kColumns));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(prompt, 0,
                       prompt_rows * kColumns * sizeof(__nv_bfloat16));
  }
  if (error != cudaSuccess) return fail("MMQ allocation", error);

  for (unsigned int tile : {1U, 2U, 4U, 8U}) {
    for (int warmup = 0; warmup < kWarmups && error == cudaSuccess; ++warmup) {
      error = qw38::cuda::launch_quant_mmq_variant(
          kKind, weights, kRows, kColumns, prompt, prompt_rows, staged, output,
          tile, nullptr);
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (error == cudaSuccess) error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    for (int sample = 0; sample < kSamples && error == cudaSuccess; ++sample) {
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_quant_mmq_variant(
            kKind, weights, kRows, kColumns, prompt, prompt_rows, staged,
            output, tile, nullptr);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float milliseconds = 0.0F;
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&milliseconds, start, stop);
      }
      if (error == cudaSuccess) {
        std::printf(
            "tune=mmq kind=q4_k prompt_rows=%zu output_rows=%zu columns=%zu "
            "tile=%u sample=%d milliseconds=%.9g\n",
            prompt_rows, kRows, kColumns, tile, sample, milliseconds);
      }
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
  }
  cudaFree(output);
  cudaFree(staged);
  cudaFree(prompt);
  cudaFree(weights);
  return error == cudaSuccess ? 0 : fail("MMQ sweep", error);
}

}  // namespace

int main() {
  struct MmvCase final {
    qw38::cuda::QuantKind kind;
    std::size_t rows;
    std::size_t columns;
  };
  constexpr std::array<MmvCase, 8> kMmvCases{{
      {qw38::cuda::QuantKind::kQ8_0, 48, 5120},
      {qw38::cuda::QuantKind::kQ8_0, 1024, 5120},
      {qw38::cuda::QuantKind::kQ4K, 5120, 17408},
      {qw38::cuda::QuantKind::kQ8_0, 6144, 5120},
      {qw38::cuda::QuantKind::kQ8_0, 10240, 5120},
      {qw38::cuda::QuantKind::kQ8_0, 12288, 5120},
      {qw38::cuda::QuantKind::kQ4K, 17408, 5120},
      {qw38::cuda::QuantKind::kQ6K, 248320, 5120},
  }};
  for (const MmvCase& test : kMmvCases) {
    if (run_mmv(test.kind, test.rows, test.columns) != 0) return 1;
  }
  for (std::size_t prompt_rows : {1U, 2U, 4U, 8U, 16U, 32U, 64U}) {
    if (run_mmq(prompt_rows) != 0) return 1;
  }
  std::printf("status=passed\n");
  return 0;
}
