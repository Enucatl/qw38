#include "cuda/attention.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"
#include "src/format/floatcvt.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

// FAST-02: one complete operation per phase, no warmups or repetitions.
int main() {
  using namespace qw38::cuda;
  constexpr std::uint32_t prefix = 32768, rows = 32, capacity = prefix + rows;
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
  auto const segments = attention_partition_count(prefix + 1, prop.multiProcessorCount);
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
              << " rows=" << (prefill ? rows : 1u) << " complete_gpu_ms=" << *ms << '\n';
  }
}
