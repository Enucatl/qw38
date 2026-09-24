#include "cuda/alloc.hpp"
#include "cuda/error.hpp"
#include "format/reader.hpp"
#include "runtime/runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: qw38_bench_artifact ARTIFACT\n";
    return 2;
  }
  using Clock = std::chrono::steady_clock;
  auto const elapsed_ms = [](Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
  };

  auto const open_start = Clock::now();
  auto artifact = qw38::format::Artifact::open(argv[1]);
  auto const open_end = Clock::now();
  if (!artifact) {
    std::cerr << qw38::format::error_message(artifact.error()) << '\n';
    return 1;
  }
  if (artifact->precision().id !=
      qw38::format::PrecisionPolicyId::CandidateV1) {
    std::cerr << "artifact does not declare the candidate precision policy\n";
    return 1;
  }

  std::uint64_t payload_bytes = 0;
  std::uint64_t scale_bytes = 0;
  std::uint32_t candidate_q4 = 0;
  std::uint32_t candidate_q8 = 0;
  for (auto const& tensor : artifact->tensors()) {
    if (std::any_of(artifact->shared_bindings().begin(),
                    artifact->shared_bindings().end(),
                    [&](auto const& binding) {
                      return binding.alias_tensor_id == tensor.tensor_id;
                    })) continue;
    payload_bytes += tensor.payload.length;
    scale_bytes += tensor.scales.length;
    candidate_q4 += tensor.quantizer ==
                    qw38::format::LogicalQuantizerId::Q4G64CandidateV1;
    candidate_q8 += tensor.quantizer ==
                    qw38::format::LogicalQuantizerId::Q8G32CandidateV1;
  }
  if (candidate_q4 == 0 || candidate_q8 == 0) {
    std::cerr << "candidate quantized tensor families are missing\n";
    return 1;
  }

  auto const create_start = Clock::now();
  auto runtime = qw38::runtime::Runtime::create();
  auto const create_end = Clock::now();
  if (!runtime) {
    std::cerr << qw38::runtime::error_message(runtime.error()) << '\n';
    return 1;
  }
  std::size_t free_before = 0;
  std::size_t total = 0;
  if (auto st = qw38::cuda::check(cudaMemGetInfo(&free_before, &total),
                                  "cudaMemGetInfo(before upload)");
      !st) {
    std::cerr << qw38::cuda::error_message(st.error()) << '\n';
    return 1;
  }
  auto const upload_start = Clock::now();
  auto model = runtime->upload(*artifact);
  auto const upload_end = Clock::now();
  if (!model) {
    std::cerr << qw38::runtime::error_message(model.error()) << '\n';
    return 1;
  }
  std::size_t free_after = 0;
  if (auto st = qw38::cuda::check(cudaMemGetInfo(&free_after, &total),
                                  "cudaMemGetInfo(after upload)");
      !st) {
    std::cerr << qw38::cuda::error_message(st.error()) << '\n';
    return 1;
  }
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    std::cerr << "getrusage failed\n";
    return 1;
  }
  std::cout << "artifact_bytes=" << std::filesystem::file_size(argv[1])
            << " payload_bytes=" << payload_bytes
            << " scale_bytes=" << scale_bytes
            << " candidate_q4_tensors=" << candidate_q4
            << " candidate_q8_tensors=" << candidate_q8
            << " open_ms=" << elapsed_ms(open_start, open_end)
            << " runtime_create_ms=" << elapsed_ms(create_start, create_end)
            << " upload_ms=" << elapsed_ms(upload_start, upload_end)
            << " device_total_bytes=" << total
            << " device_free_before_bytes=" << free_before
            << " device_free_after_bytes=" << free_after
            << " model_device_bytes=" << model->device_bytes()
            << " tracked_live_device_bytes=" << qw38::cuda::live_bytes()
            << " tracked_peak_device_bytes=" << qw38::cuda::peak_live_bytes()
            << " peak_host_rss_bytes="
            << static_cast<std::uint64_t>(usage.ru_maxrss) * 1024
            << '\n';
  return 0;
}
