#include "cuda/attention.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"
#include "src/format/floatcvt.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <charconv>
#include <chrono>
#include <iostream>
#include <span>
#include <vector>

// Core diagnostic only: no output projection/residual. For the resident M1
// boundary: qw38_bench_attention_prefill --decode ARTIFACT 4096 output.f32
// FAST-02: one first-use operation per phase, no warmups or repetitions.
int main(int argc, char** argv) {
  using namespace qw38::cuda;
  unsigned visible = 32769;
  if (argc == 2 || argc == 3) {
    std::string_view arg(argv[1]);
    auto [end, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), visible);
    if (ec != std::errc{} || end != arg.end() ||
        (visible != 4096 && visible != 32768)) return 2;
  } else if (argc != 1) return 2;
  bool const short_split = argc == 3 && std::string_view(argv[2]) == "short128";
  if (argc == 3 && !short_split) return 2;
  std::uint32_t const prefix = visible - 1, rows = 32, capacity = prefix + rows;
  constexpr std::uint32_t H = kAttnQueryHeads, D = kAttnHeadDim;
  constexpr std::uint32_t W = kAttnKvHeads * D;
  auto stream = Stream::create();
  if (!stream) return 1;
  auto upload = [&](std::vector<std::uint16_t> const& values) {
    auto result = DeviceBuffer::allocate(values.size() * 2u);
    if (result && (!copy_h2d(result->data(), std::as_bytes(std::span(values)), *stream) ||
                   !stream->sync()))
      return decltype(result)(std::unexpected(make_error(
          ErrorCode::InvalidArgument, "benchmark", "upload failed")));
    return result;
  };
  auto data = [](std::size_t count, unsigned seed) {
    std::vector<std::uint16_t> result(count);
    for (std::size_t i = 0; i < count; ++i)
      result[i] = qw38::format::fp32_to_bf16_rne(
          0.01f * (static_cast<int>((i * seed + i / D) % 101) - 50));
    return result;
  };
  auto qg = upload(data(rows * H * 2u * D, 7));
  auto k = upload(data(rows * W, 11));
  auto v = upload(data(rows * W, 13));
  auto gamma = upload(std::vector<std::uint16_t>(2u * D, 0));
  auto kv = upload(data(16ull * 2 * W * capacity, 17));
  auto q = DeviceBuffer::allocate(rows * H * D * 2u);
  auto g = DeviceBuffer::allocate(rows * H * D * 2u);
  auto y = DeviceBuffer::allocate(rows * H * D * 2u);
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, stream->device()) != cudaSuccess) return 1;
  auto const segments = short_split && visible <= 4096
      ? std::min((visible + 127) / 128, attention_partition_count(capacity * 2, prop.multiProcessorCount))
      : attention_partition_count(visible, prop.multiProcessorCount);
  auto partials = DeviceBuffer::allocate(H * segments * kAttnPartialStride * 4u);
  auto frequencies = DeviceBuffer::allocate(kAttnRopeFreqs * 4u);
  auto start = Event::create_timing();
  auto end = Event::create_timing();
  if (!qg || !k || !v || !gamma || !kv || !q || !g || !y || !partials ||
      !frequencies || !start || !end || !zero(*frequencies, *stream) ||
      !stream->sync()) return 1;
  auto resources = attention_prefill_resources();
  auto decode_resources = attention_decode_resources();
  if (!resources || !decode_resources) return 1;
  std::cout << "device=" << prop.name << " prefix=" << prefix
            << " capacity=" << capacity << " warmups=0 repetitions=1"
            << " registers=" << resources->registers
            << " shared_bytes=" << resources->shared_bytes
            << " local_bytes=" << resources->local_bytes
            << " blocks_per_sm=" << resources->occupancy_blocks_per_sm << '\n';
  std::cout << "decode_partitions=" << segments << " sm_count=" << prop.multiProcessorCount
            << " workspace_bytes=" << H * segments * kAttnPartialStride * 4u
            << " registers=" << decode_resources->registers
            << " shared_bytes=" << decode_resources->shared_bytes
            << " local_bytes=" << decode_resources->local_bytes
            << " blocks_per_sm=" << decode_resources->occupancy_blocks_per_sm << '\n';
  for (bool prefill : {true, false}) {
    if (argc >= 2 && prefill) continue;
    auto const host_start = std::chrono::steady_clock::now();
    if (!start->record(*stream) || !launch_attention_prepare_chunk(
        static_cast<std::uint16_t const*>(qg->data()),
        static_cast<std::uint16_t const*>(k->data()),
        static_cast<std::uint16_t const*>(v->data()),
        static_cast<std::uint16_t const*>(gamma->data()) + D,
        static_cast<std::uint16_t const*>(gamma->data()),
        static_cast<float const*>(frequencies->data()), 1e-6f,
        prefix, prefill ? rows : 1u, static_cast<std::uint16_t*>(q->data()),
        static_cast<std::uint16_t*>(g->data()), static_cast<std::uint16_t*>(kv->data()),
        0, capacity, *stream)) return 1;
    if (prefill) {
      if (!launch_attention_prefill_scan(
          static_cast<std::uint16_t const*>(q->data()),
          static_cast<std::uint16_t const*>(g->data()),
          static_cast<std::uint16_t const*>(kv->data()), 0, capacity, prefix,
          rows, static_cast<std::uint16_t*>(y->data()), *stream)) return 1;
    } else if (!launch_attention_scan(
        static_cast<std::uint16_t const*>(q->data()),
        static_cast<std::uint16_t const*>(kv->data()), 0, capacity, prefix + 1,
        static_cast<float*>(partials->data()), segments, *stream) ||
        !launch_attention_merge(static_cast<float const*>(partials->data()),
        static_cast<std::uint16_t const*>(g->data()), segments,
        static_cast<std::uint16_t*>(y->data()), *stream)) return 1;
    if (!end->record(*stream) || !end->sync()) return 1;
    auto ms = elapsed_ms(*start, *end);
    if (!ms) return 1;
    std::cout << "phase=" << (prefill ? "prefill" : "decode")
              << " rows=" << (prefill ? rows : 1u) << " gpu_event_interval_ms=" << *ms
              << " timing_schema=2 boundary=prepare_append_scan_gate_core_only"
              << " first_use=true readout_included=false host_ms=" << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - host_start).count() << '\n';
  }
}
