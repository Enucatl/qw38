#include "cuda/attention.hpp"
#include "cuda/buffer.hpp"
#include "cuda/error.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using qw38::cuda::DeviceBuffer;
using qw38::cuda::Event;
using qw38::cuda::Stream;
using qw38::cuda::elapsed_ms;

namespace {

bool launch_once(DeviceBuffer& q, DeviceBuffer& kv, DeviceBuffer& partials,
                 DeviceBuffer& g, DeviceBuffer& y, std::uint64_t capacity,
                 std::uint64_t populated, std::uint32_t segments,
                 Stream const& stream) {
  return qw38::cuda::zero(partials, stream) &&
         qw38::cuda::launch_attention_scan(
             static_cast<std::uint16_t const*>(q.data()),
             static_cast<std::uint16_t const*>(kv.data()), 0, capacity,
             populated, static_cast<float*>(partials.data()), segments,
             stream) &&
         qw38::cuda::launch_attention_merge(
             static_cast<float const*>(partials.data()),
             static_cast<std::uint16_t const*>(g.data()), segments,
             static_cast<std::uint16_t*>(y.data()), stream);
}

float percentile(std::vector<float> samples, float p) {
  std::sort(samples.begin(), samples.end());
  float const rank = p * static_cast<float>(samples.size() - 1u);
  auto const lo = static_cast<std::size_t>(rank);
  auto const hi = std::min(lo + 1u, samples.size() - 1u);
  float const frac = rank - static_cast<float>(lo);
  return samples[lo] + frac * (samples[hi] - samples[lo]);
}

float standard_error(std::vector<float> const& samples) {
  float mean = 0.0f;
  for (float const sample : samples) mean += sample;
  mean /= static_cast<float>(samples.size());
  float variance = 0.0f;
  for (float const sample : samples) {
    float const delta = sample - mean;
    variance += delta * delta;
  }
  variance /= static_cast<float>(samples.size() - 1u);
  return std::sqrt(variance / static_cast<float>(samples.size()));
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  if (auto st = qw38::cuda::check(cudaGetDeviceProperties(&prop, 0),
                                  "cudaGetDeviceProperties");
      !st) {
    std::cerr << qw38::cuda::error_message(st.error()) << '\n';
    return 1;
  }
  auto stream = Stream::create();
  if (!stream) return 1;

  constexpr std::uint64_t capacity = 32768;
  constexpr std::uint64_t kv_bytes =
      16ull * 2ull * qw38::cuda::kAttnKvHeads * capacity *
      qw38::cuda::kAttnHeadDim * sizeof(std::uint16_t);
  constexpr std::uint64_t partial_bytes =
      static_cast<std::uint64_t>(qw38::cuda::kAttnQueryHeads) *
      ((capacity + 255u) / 256u) * qw38::cuda::kAttnPartialStride *
      sizeof(float);
  auto q = DeviceBuffer::allocate(
      static_cast<std::uint64_t>(qw38::cuda::kAttnQueryHeads) *
      qw38::cuda::kAttnHeadDim * sizeof(std::uint16_t));
  auto g = DeviceBuffer::allocate(
      static_cast<std::uint64_t>(qw38::cuda::kAttnQueryHeads) *
      qw38::cuda::kAttnHeadDim * sizeof(std::uint16_t));
  auto kv = DeviceBuffer::allocate(kv_bytes);
  auto partials = DeviceBuffer::allocate(partial_bytes);
  auto y = DeviceBuffer::allocate(
      static_cast<std::uint64_t>(qw38::cuda::kAttnQueryHeads) *
      qw38::cuda::kAttnHeadDim * sizeof(std::uint16_t));
  if (!q || !g || !kv || !partials || !y) {
    std::cerr << "attention benchmark allocation failed\n";
    return 1;
  }
  if (!qw38::cuda::zero(*q, *stream) || !qw38::cuda::zero(*g, *stream) ||
      !qw38::cuda::zero(*kv, *stream)) {
    return 1;
  }

  std::cout << "qw38_bench_attention\n";
  std::cout << "mode=diagnostic-only\n";
  std::cout << "device=" << prop.name << " sm_" << prop.major << prop.minor
            << " capacity=" << capacity << " kv_bytes=" << kv_bytes << '\n';
  std::cout << "geometry threads=" << qw38::cuda::kAttnScanThreads
            << " segment_keys=" << qw38::cuda::kAttnSegmentKeys
            << " subtile_keys=" << qw38::cuda::kAttnSubtileKeys
            << " merge_threads=" << qw38::cuda::kAttnMergeThreads << '\n';

  for (std::uint64_t populated : {512ull, 4096ull, 32768ull}) {
    auto const segments = static_cast<std::uint32_t>((populated + 255u) / 256u);
    constexpr int warmup = 5;
    constexpr int repetitions = 20;
    for (int i = 0; i < warmup; ++i) {
      if (!launch_once(*q, *kv, *partials, *g, *y, capacity, populated,
                       segments, *stream)) {
        return 1;
      }
    }
    std::vector<float> samples;
    samples.reserve(repetitions);
    for (int i = 0; i < repetitions; ++i) {
      auto t0 = Event::create_timing();
      auto t1 = Event::create_timing();
      if (!t0 || !t1 || !t0->record(*stream) ||
          !launch_once(*q, *kv, *partials, *g, *y, capacity, populated,
                       segments, *stream) ||
          !t1->record(*stream) || !t1->sync()) {
        return 1;
      }
      auto ms = elapsed_ms(*t0, *t1);
      if (!ms) return 1;
      samples.push_back(*ms);
    }
    std::cout << "populated=" << populated << " segments=" << segments
              << " warmups=" << warmup << " repetitions=" << repetitions
              << " median_ms=" << percentile(samples, 0.50f)
              << " p99_ms=" << percentile(samples, 0.99f)
              << " uncertainty_ms=" << standard_error(samples) << '\n';
  }
  return 0;
}
