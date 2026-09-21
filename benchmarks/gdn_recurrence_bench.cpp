#include "cuda/buffer.hpp"
#include "cuda/error.hpp"
#include "cuda/event.hpp"
#include "cuda/gdn.hpp"
#include "cuda/stream.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

using qw38::cuda::DeviceBuffer;
using qw38::cuda::Event;
using qw38::cuda::Stream;
using qw38::cuda::elapsed_ms;
using qw38::cuda::gdn_recurrence_resources;
using qw38::cuda::kGdnKeyHeads;
using qw38::cuda::kGdnRecurBlocksPerLayer;
using qw38::cuda::kGdnRecurThreads;
using qw38::cuda::kGdnRecurWarps;
using qw38::cuda::kGdnSElemsPerLayer;
using qw38::cuda::kGdnValueHeads;
using qw38::cuda::launch_gdn_recurrence;
using qw38::cuda::zero;

int main() {
  cudaDeviceProp prop{};
  if (auto st = qw38::cuda::check(cudaGetDeviceProperties(&prop, 0),
                                  "cudaGetDeviceProperties");
      !st) {
    std::cerr << qw38::cuda::error_message(st.error()) << '\n';
    return 1;
  }
  auto stream = Stream::create();
  if (!stream) {
    std::cerr << qw38::cuda::error_message(stream.error()) << '\n';
    return 1;
  }

  auto res = gdn_recurrence_resources();
  if (!res) {
    std::cerr << qw38::cuda::error_message(res.error()) << '\n';
    return 1;
  }

  std::uint64_t const nqk = static_cast<std::uint64_t>(kGdnKeyHeads) * 128u;
  std::uint64_t const nv = static_cast<std::uint64_t>(kGdnValueHeads) * 128u;
  auto q = DeviceBuffer::allocate(nqk * 4);
  auto k = DeviceBuffer::allocate(nqk * 4);
  auto alpha = DeviceBuffer::allocate(kGdnValueHeads * 4);
  auto beta = DeviceBuffer::allocate(kGdnValueHeads * 4);
  auto v = DeviceBuffer::allocate(nv * 2);
  auto s = DeviceBuffer::allocate(static_cast<std::uint64_t>(kGdnSElemsPerLayer) * 4);
  auto o = DeviceBuffer::allocate(nv * 4);
  if (!q || !k || !alpha || !beta || !v || !s || !o) {
    std::cerr << "allocate failed\n";
    return 1;
  }
  if (!zero(*q, *stream) || !zero(*k, *stream) || !zero(*alpha, *stream) ||
      !zero(*beta, *stream) || !zero(*v, *stream) || !zero(*s, *stream) ||
      !zero(*o, *stream)) {
    std::cerr << "zero failed\n";
    return 1;
  }

  int const warmup = 2;
  int const launches = 8;
  for (int i = 0; i < warmup; ++i) {
    auto st = launch_gdn_recurrence(
        static_cast<float const*>(q->data()), static_cast<float const*>(k->data()),
        static_cast<float const*>(alpha->data()),
        static_cast<float const*>(beta->data()),
        static_cast<std::uint16_t const*>(v->data()), static_cast<float*>(s->data()),
        0, static_cast<float*>(o->data()), *stream);
    if (!st) {
      std::cerr << qw38::cuda::error_message(st.error()) << '\n';
      return 1;
    }
  }
  auto t0 = Event::create_timing();
  auto t1 = Event::create_timing();
  if (!t0 || !t1) {
    return 1;
  }
  auto r0 = t0->record(*stream);
  if (!r0) {
    return 1;
  }
  for (int i = 0; i < launches; ++i) {
    auto st = launch_gdn_recurrence(
        static_cast<float const*>(q->data()), static_cast<float const*>(k->data()),
        static_cast<float const*>(alpha->data()),
        static_cast<float const*>(beta->data()),
        static_cast<std::uint16_t const*>(v->data()), static_cast<float*>(s->data()),
        0, static_cast<float*>(o->data()), *stream);
    if (!st) {
      std::cerr << qw38::cuda::error_message(st.error()) << '\n';
      return 1;
    }
  }
  auto r1 = t1->record(*stream);
  if (!r1) {
    return 1;
  }
  auto s1 = t1->sync();
  if (!s1) {
    return 1;
  }
  auto ms = elapsed_ms(*t0, *t1);
  if (!ms) {
    std::cerr << qw38::cuda::error_message(ms.error()) << '\n';
    return 1;
  }
  float const avg = *ms / static_cast<float>(launches);
  std::uint64_t const s_bytes =
      static_cast<std::uint64_t>(kGdnSElemsPerLayer) * 4u;
  std::uint64_t const traffic = s_bytes * 2u + nqk * 8u + kGdnValueHeads * 8u +
                                nv * 2u + nv * 4u;

  std::cout << "qw38_bench_gdn_recurrence\n";
  std::cout << "device=" << prop.name << " sm_" << prop.major << prop.minor << '\n';
  std::cout << "geometry warps/block=" << kGdnRecurWarps
            << " threads=" << kGdnRecurThreads
            << " blocks/layer=" << kGdnRecurBlocksPerLayer
            << " ownership=warp-per-value HVK\n";
  std::cout << "resources registers=" << res->registers
            << " shared_bytes=" << res->shared_bytes
            << " local_bytes=" << res->local_bytes
            << " const_bytes=" << res->const_bytes
            << " max_threads=" << res->max_threads_per_block
            << " occupancy_blocks_per_sm=" << res->occupancy_blocks_per_sm << '\n';
  std::cout << "one-step s_bytes=" << s_bytes << " traffic_bytes~=" << traffic
            << " launches=" << launches << " ms=" << avg << '\n';
  return 0;
}
