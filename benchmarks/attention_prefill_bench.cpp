#include "runtime/prefill.hpp"
#include "runtime/runtime.hpp"

#include "cuda/alloc.hpp"
#include "cuda/attention.hpp"
#include "cuda/copy.hpp"
#include "cuda/event.hpp"
#include "cuda/upload.hpp"
#include "format/floatcvt.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {
template <typename T>
std::span<std::byte const> bytes(std::vector<T> const& v) {
  return {reinterpret_cast<std::byte const*>(v.data()), v.size() * sizeof(T)};
}

int run(char const* artifact) {
  using namespace qw38::runtime;
  auto runtime = Runtime::create();
  if (!runtime) { std::cerr << error_message(runtime.error()) << '\n'; return 1; }
  auto model = runtime->load(artifact);
  if (!model) { std::cerr << error_message(model.error()) << '\n'; return 1; }
  constexpr std::uint32_t capacity = 32768u + 32u, m = 32u;
  auto session = runtime->create_session(*model, capacity);
  if (!session) { std::cerr << error_message(session.error()) << '\n'; return 1; }
  auto const& stream = runtime->stream();
  auto plan = bind_prefill_attention_layer(*model, *session, 3, stream);
  auto engine = qw38::cuda::PrefillEngine::create(stream, m);
  auto workspace = PrefillAttentionWorkspace::create(m, stream.device());
  auto resources = qw38::cuda::attention_prefill_resources();
  auto control_resources = qw38::cuda::attention_prefill_resources(4);
  if (!plan || !engine || !workspace || !resources || !control_resources) return 1;
  cudaDeviceProp gpu{};
  if (!qw38::cuda::check(cudaGetDeviceProperties(&gpu, stream.device()),
                         "cudaGetDeviceProperties")) return 1;
  std::size_t free_bytes = 0, total_bytes = 0;
  if (!qw38::cuda::check(cudaMemGetInfo(&free_bytes, &total_bytes),
                         "cudaMemGetInfo")) return 1;
  auto embedding = model->payload("model.language_model.embed_tokens.weight");
  if (!embedding) return 1;
  std::vector<std::uint16_t> embedding_rows(m * kHidden);
  auto const* table = static_cast<std::uint16_t const*>(embedding->pointer);
  for (std::uint32_t t = 0; t < m; ++t)
    if (!qw38::cuda::copy_d2h(embedding_rows.data() + t * kHidden,
        table + static_cast<std::uint64_t>(3000u + t * 17u) * kHidden,
        kHidden * 2u, stream)) return 1;
  if (!stream.sync()) return 1;
  std::vector<float> input(m * kHidden);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = qw38::format::bf16_to_fp32(embedding_rows[i]);
  auto din = qw38::cuda::upload(bytes(input), stream);
  auto dmid = qw38::cuda::DeviceBuffer::allocate(input.size() * 4u, stream.device());
  auto dout = qw38::cuda::DeviceBuffer::allocate(input.size() * 4u, stream.device());
  auto start = qw38::cuda::Event::create_timing();
  auto end = qw38::cuda::Event::create_timing();
  if (!din || !dmid || !dout || !start || !end) return 1;
  auto s = workspace->slices();
  auto const allocation_mark = qw38::cuda::malloc_count();
  std::uint64_t const kv_bytes =
      16ull * 2ull * kKvHeads * capacity * kHeadDim * 2ull;
  std::cout << "context,gpu," << gpu.name << ",sm," << gpu.major << gpu.minor
            << ",artifact_device_bytes," << model->device_bytes()
            << ",workspace_bytes," << workspace->bytes() + engine->workspace_bytes()
            << ",kv_bytes," << kv_bytes << ",free_bytes," << free_bytes
            << ",total_bytes," << total_bytes
            << ",query_tile," << qw38::cuda::kAttnPrefillQueryTile
            << ",key_tile," << qw38::cuda::kAttnPrefillKeyTile
            << ",registers," << resources->registers
            << ",shared_bytes," << resources->shared_bytes
            << ",local_bytes," << resources->local_bytes
            << ",occupancy_blocks_per_sm," << resources->occupancy_blocks_per_sm
            << '\n';
  std::cout << "control,query_tile,4,key_tile," << qw38::cuda::kAttnPrefillKeyTile
            << ",registers," << control_resources->registers
            << ",shared_bytes," << control_resources->shared_bytes
            << ",local_bytes," << control_resources->local_bytes
            << ",occupancy_blocks_per_sm," << control_resources->occupancy_blocks_per_sm
            << '\n';
  std::cout << "sample,kind,prefix,m,query_tile,rep,gpu_ms,kv_unique_read_bytes,kv_read_estimate_bytes\n";
  for (std::uint32_t prefix : {512u, 4096u, 32768u}) {
    if (!session->reset() || !session->set_populated_length(0, prefix)) return 1;
    auto warmup = execute_prefill_attention_layer(*plan, *workspace, *engine,
        static_cast<float const*>(din->data()), static_cast<float*>(dmid->data()),
        static_cast<float*>(dout->data()), m, prefix);
    if (!warmup) { std::cerr << error_message(warmup.error()) << '\n'; return 1; }
    for (std::uint32_t rep = 0; rep < 3u; ++rep) {
      if (!session->reset() || !session->set_populated_length(0, prefix)) return 1;
      if (prefix != 32768u) {
        if (!start->record(stream)) return 1;
        auto st = execute_prefill_attention_layer(*plan, *workspace, *engine,
            static_cast<float const*>(din->data()), static_cast<float*>(dmid->data()),
            static_cast<float*>(dout->data()), m, prefix);
        if (!st) { std::cerr << error_message(st.error()) << '\n'; return 1; }
        if (!end->record(stream) || !end->sync()) return 1;
        auto ms = qw38::cuda::elapsed_ms(*start, *end);
        if (!ms) return 1;
        std::uint64_t const traffic = 2ull * kKvHeads * (prefix + m) * kHeadDim * 2u;
        std::uint64_t estimated = 0;
        for (std::uint32_t q = 0; q < m; q += 1u)
          estimated += 2ull * kQueryHeads * (prefix + std::min(m, q + 1u)) *
                       kHeadDim * 2u;
        std::cout << "sample,complete_layer," << prefix << ',' << m << ",1," << rep
                  << ',' << *ms << ',' << traffic << ',' << estimated << '\n';
      } else {
        // A real projection/preparation pass supplies Q/g and valid appended KV.
        auto st = execute_prefill_attention_layer(*plan, *workspace, *engine,
            static_cast<float const*>(din->data()), static_cast<float*>(dmid->data()),
            static_cast<float*>(dout->data()), m, prefix);
        if (!st) { std::cerr << error_message(st.error()) << '\n'; return 1; }
      }
      for (std::uint32_t tile : {1u, 4u}) {
        if (!start->record(stream) ||
            !qw38::cuda::launch_attention_prefill_scan(
                s.q, s.g, static_cast<std::uint16_t const*>(session->kv().pointer),
                0, capacity, prefix, m, s.y, stream, tile) ||
            !end->record(stream) || !end->sync()) return 1;
        auto ms = qw38::cuda::elapsed_ms(*start, *end);
        if (!ms || qw38::cuda::malloc_count() != allocation_mark) return 1;
        std::uint64_t const traffic = 2ull * kKvHeads * (prefix + m) * kHeadDim * 2u;
        std::uint64_t estimated = 0;
        for (std::uint32_t q = 0; q < m; q += tile)
          estimated += 2ull * kQueryHeads * (prefix + std::min(m, q + tile)) *
                       kHeadDim * 2u;
        std::cout << "sample,attention_only," << prefix << ',' << m << ',' << tile
                  << ',' << rep << ',' << *ms << ',' << traffic << ',' << estimated
                  << '\n';
      }
    }
  }
  return 0;
}
}  // namespace

int main() {
  auto const* artifact = std::getenv("QW38_AUTHORITY_ARTIFACT");
  if (!artifact) { std::cerr << "QW38_AUTHORITY_ARTIFACT is required\n"; return 1; }
  return run(artifact);
}
