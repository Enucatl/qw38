#include "runtime/prefill.hpp"
#include "runtime/runtime.hpp"

#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "cuda/event.hpp"
#include "cuda/gdn.hpp"
#include "cuda/upload.hpp"
#include "format/floatcvt.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {

template <class T>
std::span<std::byte const> bytes(std::vector<T> const& v) {
  return {reinterpret_cast<std::byte const*>(v.data()), v.size() * sizeof(T)};
}

int run(char const* artifact) {
  using namespace qw38::runtime;
  auto runtime = Runtime::create();
  if (!runtime) { std::cerr << error_message(runtime.error()) << '\n'; return 1; }
  auto model = runtime->load(artifact);
  if (!model) { std::cerr << error_message(model.error()) << '\n'; return 1; }
  auto session = runtime->create_session(*model, 256);
  if (!session) { std::cerr << error_message(session.error()) << '\n'; return 1; }
  auto const& stream = runtime->stream();
  auto plan = bind_prefill_gdn_layer(*model, *session, 0, stream);
  auto engine = qw38::cuda::PrefillEngine::create(stream, 256);
  auto workspace = PrefillGdnWorkspace::create(256, stream.device());
  auto resources = qw38::cuda::gdn_prefill_recurrence_resources();
  if (!plan || !engine || !workspace || !resources) return 1;
  cudaDeviceProp gpu{};
  if (!qw38::cuda::check(cudaGetDeviceProperties(&gpu, stream.device()),
                         "cudaGetDeviceProperties")) return 1;
  std::size_t free_bytes = 0, total_bytes = 0;
  if (!qw38::cuda::check(cudaMemGetInfo(&free_bytes, &total_bytes),
                         "cudaMemGetInfo")) return 1;
  auto embedding = model->payload("model.language_model.embed_tokens.weight");
  if (!embedding) return 1;
  std::vector<std::uint16_t> embedding_rows(16u * kHidden);
  auto const* table = static_cast<std::uint16_t const*>(embedding->pointer);
  for (std::uint32_t i = 0; i < 16u; ++i)
    if (!qw38::cuda::copy_d2h(embedding_rows.data() + i * kHidden,
        table + static_cast<std::uint64_t>(1000u + i * 17u) * kHidden,
        kHidden * 2u, stream)) return 1;
  if (!stream.sync()) return 1;
  std::vector<float> input(256u * kHidden);
  for (std::uint32_t t = 0; t < 256u; ++t)
    for (std::uint32_t j = 0; j < kHidden; ++j)
      input[static_cast<std::size_t>(t) * kHidden + j] =
          qw38::format::bf16_to_fp32(embedding_rows[
              static_cast<std::size_t>(t % 16u) * kHidden + j]);
  auto din = qw38::cuda::upload(bytes(input), stream);
  auto dmid = qw38::cuda::DeviceBuffer::allocate(input.size() * 4u, stream.device());
  auto dout = qw38::cuda::DeviceBuffer::allocate(input.size() * 4u, stream.device());
  auto start = qw38::cuda::Event::create_timing();
  auto end = qw38::cuda::Event::create_timing();
  if (!din || !dmid || !dout || !start || !end) return 1;
  auto const* in = static_cast<float const*>(din->data());
  auto* mid = static_cast<float*>(dmid->data());
  auto* out = static_cast<float*>(dout->data());
  auto s = workspace->slices();
  auto const allocation_mark = qw38::cuda::malloc_count();
  std::cout << "context,gpu," << gpu.name << ",sm," << gpu.major << gpu.minor
            << ",artifact_device_bytes," << model->device_bytes()
            << ",workspace_bytes," << workspace->bytes() + engine->workspace_bytes()
            << ",free_bytes," << free_bytes
            << ",total_bytes," << total_bytes
            << ",recurrence_registers," << resources->registers
            << ",recurrence_shared_bytes," << resources->shared_bytes
            << ",recurrence_local_bytes," << resources->local_bytes
            << ",recurrence_occupancy_blocks_per_sm," << resources->occupancy_blocks_per_sm
            << '\n';
  std::cout << "sample,kind,m,interval,rep,gpu_ms,host_ms,state_read_write_bytes,workspace_bytes\n";
  for (std::uint32_t m : {64u, 128u, 256u}) {
    for (std::uint32_t interval : {1u, 64u, 128u, 256u}) {
      if (interval > m) continue;
      for (std::uint32_t rep = 0; rep < 5u; ++rep) {
        if (!session->reset() || !start->record(stream)) return 1;
        auto const t0 = std::chrono::steady_clock::now();
        auto st = execute_prefill_gdn_layer(*plan, *workspace, *engine,
                                            in, mid, out, m, 0, interval);
        auto const t1 = std::chrono::steady_clock::now();
        if (!st) { std::cerr << error_message(st.error()) << '\n'; return 1; }
        if (!end->record(stream) || !end->sync()) return 1;
        auto ms = qw38::cuda::elapsed_ms(*start, *end);
        if (!ms || qw38::cuda::malloc_count() != allocation_mark) return 1;
        std::uint64_t const traffic =
            ((static_cast<std::uint64_t>(m) + interval - 1u) / interval) *
            2u * kGdnSBytesPerLayer;
        std::cout << "sample,complete_layer," << m << ',' << interval << ','
                  << rep << ',' << *ms << ','
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << ',' << traffic << ','
                  << workspace->bytes() + engine->workspace_bytes() << '\n';
      }
      // The prepared token-major arrays above remain valid after a state reset.
      for (std::uint32_t rep = 0; rep < 5u; ++rep) {
        if (!session->reset() || !start->record(stream)) return 1;
        auto const t0 = std::chrono::steady_clock::now();
        auto st = qw38::cuda::launch_gdn_prefill_recurrence(
            s.q_hat, s.k_hat, s.alpha, s.beta, s.convolved,
            static_cast<float*>(session->gdn_s().pointer), 0, s.o,
            m, interval, stream);
        if (!st || !end->record(stream) || !end->sync()) return 1;
        auto const t1 = std::chrono::steady_clock::now();
        auto ms = qw38::cuda::elapsed_ms(*start, *end);
        if (!ms || qw38::cuda::malloc_count() != allocation_mark) return 1;
        std::uint64_t const traffic =
            ((static_cast<std::uint64_t>(m) + interval - 1u) / interval) *
            2u * kGdnSBytesPerLayer;
        std::cout << "sample,recurrence," << m << ',' << interval << ','
                  << rep << ',' << *ms << ','
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << ',' << traffic << ',' << workspace->bytes() << '\n';
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: qw38_bench_gdn_prefill ARTIFACT\n";
    return 2;
  }
  return run(argv[1]);
}
